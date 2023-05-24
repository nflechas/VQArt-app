from transformers import BertTokenizer, BertForQuestionAnswering
import torch

class QA(object):
    def __init__(self, 
                 model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'):
      
        self.model_name = model_name
        
        self.__load_model_and_tokenizer()
    
    def __load_model_and_tokenizer(self):
        self.model = BertForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def __get_segment_ids(self, input_ids):
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(self.tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input_ids)
        
        return segment_ids

    def answer_question(self, query, passage):
        input_ids = self.tokenizer.encode(query, passage)
        segment_ids = self.__get_segment_ids(input_ids)

        # Run our example through the model.
        outputs = self.model(torch.tensor([input_ids]), # The tokens representing our input text.
                            token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                            return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Find the tokens with the highest `start` and `end` scores.
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        return self.tokenizer.decode(input_ids[answer_start:answer_end+1])
