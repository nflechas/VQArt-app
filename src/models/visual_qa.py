from transformers import pipeline


class VisualQA(object):
    def __init__(self, model_name='nflechas/VQArt', tokenizer_name='dandelin/vilt-b32-finetuned-vqa'):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.__load_model()

    def __load_model(self):
        self.model = pipeline('vqa', model=self.model_name, tokenizer=self.tokenizer_name)

    def answer_question(self, query, image):
        return self.model(question=query, image=image, top_k=1)[0]['answer']
