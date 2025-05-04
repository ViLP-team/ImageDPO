from __future__ import annotations

from cog import BasePredictor, ConcatenateIterator, Input, Path


class VLM_wrapper(BasePredictor):
    def __init__(
        self,
        model_type="llava",
        checkpoint_path="liuhaotian/llava-v1.5-7b",
        model_name="llava-v1.5-7b",
        conv_mode="llama_3",
    ):
        # because the very messy environment between different vlms, we only import them if necessary
        if model_type == "llava":
            from vilp.model_wrapper.llava_wrapper import Predictor

            self.predictor = Predictor()
            self.predictor.setup(checkpoint_path, model_name)
        elif model_type == "cambrian":
            raise NotImplementedError("Cambrian is not implemented yet")
            # from vilp.model_wrapper.cambrian_wrapper import Cambrian_Predictor

            # self.predictor = Cambrian_Predictor()
            # self.predictor.setup(checkpoint_path, model_name, conv_mode)
        else:
            raise ValueError("Unknown model type")

    def predict(
        self,
        image: list[Path] | Path = Input(description="Input image"),
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
    ) -> ConcatenateIterator[str]:
        return self.predictor.predict(
            image, prompt, top_p, temperature, max_tokens, do_sample, num_beams
        )

    def predict_nostreaming(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
        use_bf16: bool = False,
        parallel: bool = False,
    ) -> str:
        return self.predictor.predict_nostreaming(
            image,
            prompt,
            top_p,
            temperature,
            max_tokens,
            do_sample,
            num_beams,
            use_bf16,
            parallel,
        )

    def predict_textonly(
        self,
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
    ) -> ConcatenateIterator[str]:
        return self.predictor.predict_textonly(
            prompt, top_p, temperature, max_tokens, do_sample, num_beams
        )

    def predict_text_only_nostreaming(
        self,
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
    ) -> ConcatenateIterator[str]:
        return self.predictor.predict_text_only_nostreaming(
            prompt, top_p, temperature, max_tokens
        )

    def predict_multiple_images(
        self,
        image: list[Path] | Path = Input(description="Input image"),
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
    ) -> ConcatenateIterator[str]:
        return self.predictor.predict_multiple_images(
            image, prompt, top_p, temperature, max_tokens
        )
