import json
import logging
from typing import Any, Dict, List, Union, Tuple, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.fields import Field, TextField, IndexField, LabelField, ListField, \
                                 MetadataField, SequenceLabelField, SpanField, ArrayField
from drop_bert.drop_reader import DropReaderOrg

logger = logging.getLogger(__name__)


@DatasetReader.register("squad_reader")
class SquadReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```SpacyTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    skip_invalid_examples: ``bool``, optional (default=False)
        if this is true, we will skip those invalid examples
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_invalid_examples: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)
                
                passage_tokens = []
                for token in tokenized_paragraph:
                    wordpieces = self._tokenizer.tokenize(token.text)
                    passage_tokens += wordpieces

                for question_answer in paragraph_json["qas"]:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer["text"] for answer in question_answer["answers"]]
                    span_starts = [answer["answer_start"] for answer in question_answer["answers"]]
                    span_ends = [
                        start + len(answer) for start, answer in zip(span_starts, answer_texts)
                    ]
                    additional_metadata = {"id": question_answer.get("id", None)}
                    instance = self.text_to_instance(
                        question_text, # question_text
                        paragraph, # passage_text
                        passage_tokens, # passage_tokens
                        [], # numbers_in_passage
                        [], # number_words
                        [], # number_indices
                        [] # number_len
                    )
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(self, 
                         question_text: str, 
                         passage_text: str,
                         passage_tokens: List[Token],
                         numbers_in_passage: List[Any],
                         number_words : List[str],
                         number_indices: List[int],
                         number_len: List[int],
                         question_id: str = None,
                         answer_annotations: List[Dict] = None,
                         dataset: str = None,
                         skipEmpty: bool = True) -> Union[Instance, None]:
        # Tokenize question and passage
        question_tokens = self.tokenizer.tokenize(question_text)
        qlen = len(question_tokens)
        plen = len(passage_tokens)

        question_passage_tokens = [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens
        if len(question_passage_tokens) > self.max_pieces - 1:
            question_passage_tokens = question_passage_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - qlen - 3]
            plen = len(passage_tokens)
            if len(number_indices) > 0:
                number_indices, number_len, numbers_in_passage = \
                    clipped_passage_num(number_indices, number_len, numbers_in_passage, plen)
        
        question_passage_tokens += [Token('[SEP]')]
        number_indices = [index + qlen + 2 for index in number_indices] + [-1]
        # Not done in-place so they won't change the numbers saved for the passage
        number_len = number_len + [1]
        numbers_in_passage = numbers_in_passage + [0]
        number_tokens = [Token(str(number)) for number in numbers_in_passage]
        extra_number_tokens = [Token(str(num)) for num in self.extra_numbers]
        
        mask_indices = [0, qlen + 1, len(question_passage_tokens) - 1]
        
        fields: Dict[str, Field] = {}
            
        # Add feature fields
        question_passage_field = TextField(question_passage_tokens, self.token_indexers)
        fields["question_passage"] = question_passage_field
       
        number_token_indices = \
            [ArrayField(np.arange(start_ind, start_ind + number_len[i]), padding_value=-1) 
             for i, start_ind in enumerate(number_indices)]
        fields["number_indices"] = ListField(number_token_indices)
        numbers_in_passage_field = TextField(number_tokens, self.token_indexers)
        extra_numbers_field = TextField(extra_number_tokens, self.token_indexers)
        all_numbers_field = TextField(extra_number_tokens + number_tokens, self.token_indexers)
        mask_index_fields: List[Field] = [IndexField(index, question_passage_field) for index in mask_indices]
        fields["mask_indices"] = ListField(mask_index_fields)
        
        # Compile question, passage, answer metadata
        assert self.trainDev in ('train', 'dev')
        metadata = {"original_passage": passage_text,
                    "original_question": question_text,
                    "original_numbers": numbers_in_passage,
                    "original_number_words": number_words,
                    "extra_numbers": self.extra_numbers,
                    "passage_tokens": passage_tokens,
                    "question_tokens": question_tokens,
                    "question_passage_tokens": question_passage_tokens,
                    "question_id": question_id,
                    "dataset": 'squad'}#,
                    #"iterators": self.trainIterators if self.trainDev == 'train' else self.devIterators,
                    #"trainDev": self.trainDev}
        
        
        if answer_annotations:
            for annotation in answer_annotations:
                tokenized_spans = [[token.text for token in self.tokenizer.tokenize(answer)] for answer in annotation['spans']]
                annotation['spans'] = [tokenlist_to_passage(token_list) for token_list in tokenized_spans]
            
            # Get answer type, answer text, tokenize
            answer_type, answer_texts = DropReaderOrg.extract_answer_info_from_annotation(answer_annotations[0])
            tokenized_answer_texts = []
            num_spans = min(len(answer_texts), self.max_spans)
            for answer_text in answer_texts:
                answer_tokens = self.tokenizer.tokenize(answer_text)
                tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))
            
        
            metadata["answer_annotations"] = answer_annotations
            metadata["answer_texts"] = answer_texts
            metadata["answer_tokens"] = tokenized_answer_texts
            
            # Find answer text in question and passage
            valid_question_spans = DropReaderOrg.find_valid_spans(question_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_question_spans):
                valid_question_spans[span_ind] = (span[0] + 1, span[1] + 1)
            valid_passage_spans = DropReaderOrg.find_valid_spans(passage_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_ind] = (span[0] + qlen + 2, span[1] + qlen + 2)
        
            # Get target numbers
            target_numbers = []
            for answer_text in answer_texts:
                if answer_text.strip().count(" ") == 0:
                    number = self.word_to_num(answer_text, True)
                    if number is not None:
                        target_numbers.append(number)

            # Get possible ways to arrive at target numbers with add/sub
            
            valid_expressions: List[List[int]] = []
            exp_strings = None
            if answer_type in ["number", "date"]:
                if self.exp_search == 'full':
                    expressions = get_full_exp(list(enumerate(self.extra_numbers + numbers_in_passage)),
                                               target_numbers,
                                               self.operations,
                                               self.op_dict,
                                               self.max_depth)
                    zipped = list(zip(*expressions))
                    if zipped:
                        valid_expressions = list(zipped[0])
                        exp_strings = list(zipped[1])
                elif self.exp_search == 'add_sub':
                    valid_expressions = \
                        DropReaderOrg.find_valid_add_sub_expressions(self.extra_numbers + numbers_in_passage,
                                                                  target_numbers, 
                                                                  self.max_numbers_expression)
                elif self.exp_search == 'template':
                    valid_expressions, exp_strings = \
                        get_template_exp(self.extra_numbers + numbers_in_passage, 
                                         target_numbers,
                                         self.templates,
                                         self.template_strings)
                    exp_strings = sum(exp_strings, [])
                
            
            # Get possible ways to arrive at target numbers with counting
            valid_counts: List[int] = []
            if answer_type in ["number"]:
                numbers_for_count = list(range(self.max_count + 1))
                valid_counts = DropReaderOrg.find_valid_counts(numbers_for_count, target_numbers)
            
            '''
            if len(valid_counts) == 0 and len(valid_expressions) == 0 and len(valid_question_spans) == 0 and len(valid_passage_spans) == 0:
                if skipEmpty:
                    return None
            '''
            # Update metadata with answer info
            answer_info = {"answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "num_spans": num_spans,
                           "expressions": valid_expressions,
                           "counts": valid_counts}
            if self.exp_search in ['template', 'full']:
                answer_info['expr_text'] = exp_strings
            metadata["answer_info"] = answer_info
        
            # Add answer fields
            passage_span_fields: List[Field] = [SpanField(span[0], span[1], question_passage_field) for span in valid_passage_spans]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = [SpanField(span[0], span[1], question_passage_field) for span in valid_question_spans]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)
            
            if self.exp_search == 'add_sub':
                add_sub_signs_field: List[Field] = []
                extra_signs_field: List[Field] = []
                for signs_for_one_add_sub_expressions in valid_expressions:
                    extra_signs = signs_for_one_add_sub_expressions[:len(self.extra_numbers)]
                    normal_signs = signs_for_one_add_sub_expressions[len(self.extra_numbers):]
                    add_sub_signs_field.append(SequenceLabelField(normal_signs, numbers_in_passage_field))
                    extra_signs_field.append(SequenceLabelField(extra_signs, extra_numbers_field))
                if not add_sub_signs_field:
                    add_sub_signs_field.append(SequenceLabelField([0] * len(number_tokens), numbers_in_passage_field))
                if not extra_signs_field:
                    extra_signs_field.append(SequenceLabelField([0] * len(self.extra_numbers), extra_numbers_field))
                fields["answer_as_expressions"] = ListField(add_sub_signs_field)
                if self.extra_numbers:
                    fields["answer_as_expressions_extra"] = ListField(extra_signs_field)
            elif self.exp_search in ['template', 'full']:
                expression_indices = []
                for expression in valid_expressions:
                    if not expression:
                        expression.append(3 * [-1])
                    expression_indices.append(ArrayField(np.array(expression), padding_value=-1))
                if not expression_indices:
                    expression_indices = \
                        [ArrayField(np.array([3 * [-1]]), padding_value=-1) for _ in range(len(self.templates))]
                fields["answer_as_expressions"] = ListField(expression_indices)

            count_fields: List[Field] = [LabelField(count_label, skip_indexing=True) for count_label in valid_counts]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)
            fields["impossible_answer"] = LabelField(0, skip_indexing=True)
            
            #fields["num_spans"] = LabelField(num_spans, skip_indexing=True)

        else:
            fields["answer_as_passage_spans"] = ListField([SpanField(-1, -1, question_passage_field)])
            fields["answer_as_counts"] = ListField([LabelField(-1, skip_indexing=True)])
            fields["answer_as_expressions"] = ListField([SequenceLabelField([0]*len(numbers_in_passage_field), numbers_in_passage_field)])
            fields["impossible_answer"] = LabelField(1, skip_indexing=True)
            metadata["answer_annotations"] = [{'spans':["impossible"]}]
            fields["answer_as_question_spans"] = ListField([SpanField(-1, -1, question_passage_field)])

        fields["metadata"] = MetadataField(metadata)
        
        return Instance(fields)