import lightning as pl
import torch
from transformers import (
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    GenerationConfig,
    AutoTokenizer,
    TrOCRProcessor,
)
from transformers.utils import logging

logging.set_verbosity_error()


class OCRModel(pl.LightningModule):
    def __init__(
        self,
        model_name="microsoft/trocr-base-handwritten",
        model_decoder_name=None,
        DATA_SIZE=0,
        WARMUP_STEPS=0,
        NUM_EPOCHS=0,
        BATCH_SIZE=0,
        LR=0.0,
        GRAD_ACC_STEPS=0,
        FOLD=0,
        MAX_LEN=128,
    ):
        super(OCRModel, self).__init__()

        self.NUM_EPOCHS = NUM_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.GRAD_ACC_STEPS = GRAD_ACC_STEPS
        self.WARMUP_STEPS = WARMUP_STEPS
        self.DATA_SIZE = DATA_SIZE
        self.MAX_LEN = MAX_LEN
        self.FOLD = FOLD

        self.model_decoder_name = model_decoder_name
        # self.enc_to_dec_proj = nn.Linear(384, 768)
        self.processor = TrOCRProcessor.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        if model_decoder_name is not None:
            self.processor.tokenizer = AutoTokenizer.from_pretrained(
                model_decoder_name, clean_up_tokenization_spaces=True
            )
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        if model_decoder_name is not None:
            self.model.decoder = AutoModelForCausalLM.from_pretrained(
                model_decoder_name, is_decoder=True, add_cross_attention=True
            )
        self._set_params()

        self.generation_config = GenerationConfig(
            max_length=self.MAX_LEN,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            num_beams=4,
            eos_token_id=self.processor.tokenizer.sep_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    def _set_params(self):
        if self.model_decoder_name is not None:
            # set decoder config to causal lm
            self.model.config.decoder.is_decoder = True
            self.model.config.decoder.add_cross_attention = True

            # # set special tokens used for creating the decoder_input_ids from the labels
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id

        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def detect_text(self, image) -> str:
        self.model.eval()
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values, generation_config=self.generation_config
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
