from typing import List

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class XInferenceImageGenerationConfig(BaseImageGenerationConfig):
    """
    XInference image generation config

    https://inference.readthedocs.io/en/v1.1.1/reference/generated/xinference.client.handlers.ImageModelHandle.text_to_image.html#xinference.client.handlers.ImageModelHandle.text_to_image
    """

    def get_supported_openai_params(self, model: str) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Optimization: Avoid repeated .keys() and repeated linear lookups on lists
        supported_params = self.get_supported_openai_params(model)
        supported_param_set = set(supported_params)
        optional_param_keys = set(optional_params)

        for k, v in non_default_params.items():
            if k not in optional_param_keys:
                if k in supported_param_set:
                    optional_params[k] = v
                elif not drop_params:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )
        return optional_params
