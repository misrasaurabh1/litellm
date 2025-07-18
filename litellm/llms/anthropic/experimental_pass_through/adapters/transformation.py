import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockToolUse,
    ContentBlockDelta,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    MessageBlockDelta,
    MessageDelta,
    UsageDelta,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)
from litellm.types.utils import Choices, ModelResponse, Usage

from .streaming_iterator import AnthropicStreamWrapper

if TYPE_CHECKING:
    from litellm.types.llms.anthropic import ContentBlockContentBlockDict


class AnthropicAdapter:
    def __init__(self) -> None:
        pass

    def translate_completion_input_params(
        self, kwargs
    ) -> Optional[ChatCompletionRequest]:
        """
        - translate params, where needed
        - pass rest, as is
        """

        #########################################################
        # Validate required params
        #########################################################
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        if not model:
            raise ValueError(
                "Bad Request: model is required for Anthropic Messages Request"
            )
        if not messages:
            raise ValueError(
                "Bad Request: messages is required for Anthropic Messages Request"
            )

        #########################################################
        # Created Typed Request Body
        #########################################################
        request_body = AnthropicMessagesRequest(
            model=model, messages=messages, **kwargs
        )

        translated_body = (
            LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
                anthropic_message_request=request_body
            )
        )

        return translated_body

    def translate_completion_output_params(
        self, response: ModelResponse
    ) -> Optional[AnthropicMessagesResponse]:

        return LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
            response=response
        )

    def translate_completion_output_params_streaming(
        self, completion_stream: Any, model: str
    ) -> Union[AsyncIterator[bytes], None]:
        anthropic_wrapper = AnthropicStreamWrapper(
            completion_stream=completion_stream, model=model
        )
        # Return the SSE-wrapped version for proper event formatting
        return anthropic_wrapper.async_anthropic_sse_wrapper()


class LiteLLMAnthropicMessagesAdapter:
    def __init__(self):
        pass

    ### FOR [BETA] `/v1/messages` endpoint support

    def translatable_anthropic_params(self) -> List:
        """
        Which anthropic params, we need to translate to the openai format.
        """
        # Use module constant, faster than building a list per call
        return _TRANS_PARAMS

    def translate_anthropic_messages_to_openai(  # noqa: PLR0915
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
    ) -> List:
        new_messages: List[AllMessageValues] = []
        append = new_messages.append
        extend = new_messages.extend

        for m in messages:
            role = m["role"]
            if role == _MSG_ROLE_USER:
                message_content = m.get("content")
                # Fast path: ordinary user message string
                if isinstance(message_content, str):
                    append(ChatCompletionUserMessage(role=_MSG_ROLE_USER, content=message_content))
                    continue

                # If content is a list
                if not message_content:
                    continue

                user_content_list = []
                tool_message_list = []
                for content in message_content:
                    ctype = content.get("type")
                    # --- user content ---
                    if ctype == _TYPE_TEXT:
                        user_content_list.append(
                            ChatCompletionTextObject(type=_TYPE_TEXT, text=content.get("text", ""))
                        )
                    elif ctype == _TYPE_IMAGE:
                        source = content.get("source", "")
                        user_content_list.append(
                            ChatCompletionImageObject(
                                type=_TYPE_IMAGE_URL,
                                image_url=ChatCompletionImageUrlObject(
                                    url=f"data:{ctype};base64,{source}"
                                ),
                            )
                        )
                    # --- tool result as tool message(s) ---
                    elif ctype == _TYPE_TOOL_RESULT:
                        tool_use_id = content.get("tool_use_id", "")
                        c_inner = content.get("content", None)
                        # No content: make empty tool message
                        if "content" not in content:
                            tool_message_list.append(
                                ChatCompletionToolMessage(role=_MSG_ROLE_TOOL, tool_call_id=tool_use_id, content="")
                            )
                        elif isinstance(c_inner, str):
                            tool_message_list.append(
                                ChatCompletionToolMessage(role=_MSG_ROLE_TOOL, tool_call_id=tool_use_id, content=c_inner)
                            )
                        elif isinstance(c_inner, list):
                            for c in c_inner:
                                if isinstance(c, str):
                                    tool_message_list.append(
                                        ChatCompletionToolMessage(role=_MSG_ROLE_TOOL, tool_call_id=tool_use_id, content=c)
                                    )
                                elif isinstance(c, dict):
                                    ctype2 = c.get("type")
                                    if ctype2 == _TYPE_TEXT:
                                        tool_message_list.append(
                                            ChatCompletionToolMessage(
                                                role=_MSG_ROLE_TOOL,
                                                tool_call_id=tool_use_id,
                                                content=c.get("text", ""),
                                            )
                                        )
                                    elif ctype2 == _TYPE_IMAGE:
                                        source2 = c.get("source", "")
                                        image_str = f"data:{ctype2};base64,{source2}"
                                        tool_message_list.append(
                                            ChatCompletionToolMessage(
                                                role=_MSG_ROLE_TOOL,
                                                tool_call_id=tool_use_id,
                                                content=image_str,
                                            )
                                        )
                # Only append as necessary for user/tool message batching
                if tool_message_list:
                    extend(tool_message_list)
                if user_content_list:
                    append({"role": _MSG_ROLE_USER, "content": user_content_list})

            elif role == _MSG_ROLE_ASSISTANT:
                m_content = m.get("content")
                # Fast path: string
                if isinstance(m_content, str):
                    assistant_message = ChatCompletionAssistantMessage(
                        role=_MSG_ROLE_ASSISTANT,
                        content=m_content,
                    )
                    append(assistant_message)
                # List of content objects
                elif m_content and isinstance(m_content, list):
                    assistant_message_strs = []
                    tool_calls = []
                    for content in m_content:
                        if isinstance(content, str):
                            assistant_message_strs.append(content)
                        elif isinstance(content, dict):
                            ctype = content.get("type")
                            if ctype == _TYPE_TEXT:
                                txt = content.get("text", "")
                                assistant_message_strs.append(txt)
                            elif ctype == _TYPE_TOOL_USE:
                                function_chunk = ChatCompletionToolCallFunctionChunk(
                                    name=content.get("name", ""),
                                    arguments=json.dumps(content.get("input", {})),
                                )
                                tool_calls.append(
                                    ChatCompletionAssistantToolCall(
                                        id=content.get("id", ""),
                                        type=_TYPE_FUNCTION,
                                        function=function_chunk,
                                    )
                                )
                    assistant_message_str = "".join(assistant_message_strs) if assistant_message_strs else None
                    if assistant_message_str is not None or tool_calls:
                        assistant_message = ChatCompletionAssistantMessage(
                            role=_MSG_ROLE_ASSISTANT,
                            content=assistant_message_str,
                        )
                        if tool_calls:
                            assistant_message["tool_calls"] = tool_calls
                        append(assistant_message)
        return new_messages

    def translate_anthropic_tool_choice_to_openai(
        self, tool_choice: AnthropicMessagesToolChoice
    ) -> ChatCompletionToolChoiceValues:
        ttype = tool_choice["type"]
        if ttype == "any":
            return "required"
        elif ttype == "auto":
            return "auto"
        elif ttype == "tool":
            tc_function_param = ChatCompletionToolChoiceFunctionParam(
                name=tool_choice.get("name", "")
            )
            return ChatCompletionToolChoiceObjectParam(
                type=_TYPE_FUNCTION, function=tc_function_param
            )
        else:
            raise ValueError(
                "Incompatible tool choice param submitted - {}".format(tool_choice)
            )

    def translate_anthropic_tools_to_openai(
        self, tools: List[AllAnthropicToolsValues]
    ) -> List[ChatCompletionToolParam]:
        new_tools: List[ChatCompletionToolParam] = []
        for tool in tools:
            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=tool["name"],
            )
            ic_schema = tool.get("input_schema", None)
            if ic_schema is not None:
                function_chunk["parameters"] = ic_schema  # type: ignore
            descr = tool.get("description", None)
            if descr is not None:
                function_chunk["description"] = descr  # type: ignore
            # Only update with unmapped keys
            for k, v in tool.items():
                if k not in _TOOLS_MAPPED_PARAMS:
                    function_chunk.setdefault("parameters", {}).update({k: v})
            new_tools.append(
                ChatCompletionToolParam(type=_TYPE_FUNCTION, function=function_chunk)
            )
        return new_tools

    def translate_anthropic_to_openai(
        self, anthropic_message_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """
        This is used by the beta Anthropic Adapter, for translating anthropic `/v1/messages` requests to the openai format.
        """
        ## CONVERT ANTHROPIC MESSAGES TO OPENAI
        messages_list = cast(
            List[
                Union[
                    AnthropicMessagesUserMessageParam,
                    AnthopicMessagesAssistantMessageParam,
                ]
            ],
            anthropic_message_request["messages"],
        )
        new_messages = self.translate_anthropic_messages_to_openai(
            messages=messages_list
        )
        # Insert system message at the start (fast-path, only if necessary)
        system_content = anthropic_message_request.get("system", None)
        if system_content:
            new_messages.insert(
                0,
                ChatCompletionSystemMessage(role="system", content=system_content),
            )

        new_kwargs: ChatCompletionRequest = {
            "model": anthropic_message_request["model"],
            "messages": new_messages,
        }
        # Convert user_id from metadata
        metadata = anthropic_message_request.get("metadata", None)
        if metadata and "user_id" in metadata:
            new_kwargs["user"] = metadata["user_id"]

        # Pass litellm proxy specific metadata
        if "litellm_metadata" in anthropic_message_request:
            new_kwargs["metadata"] = anthropic_message_request.pop("litellm_metadata")

        # Convert tool choice
        tool_choice = anthropic_message_request.get("tool_choice", None)
        if tool_choice:
            new_kwargs["tool_choice"] = (
                self.translate_anthropic_tool_choice_to_openai(
                    tool_choice=cast(AnthropicMessagesToolChoice, tool_choice)
                )
            )
        # Convert tools
        tools = anthropic_message_request.get("tools", None)
        if tools:
            new_kwargs["tools"] = self.translate_anthropic_tools_to_openai(
                tools=cast(List[AllAnthropicToolsValues], tools)
            )

        # Pass remaining non-translated params as is
        translatable_params = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # pass remaining params as is
                new_kwargs[k] = v  # type: ignore

        return new_kwargs

    def _translate_openai_content_to_anthropic(
        self, choices: List[Choices]
    ) -> List[
        Union[AnthropicResponseContentBlockText, AnthropicResponseContentBlockToolUse]
    ]:
        new_content: List[
            Union[
                AnthropicResponseContentBlockText, AnthropicResponseContentBlockToolUse
            ]
        ] = []
        for choice in choices:
            if (
                choice.message.tool_calls is not None
                and len(choice.message.tool_calls) > 0
            ):
                for tool_call in choice.message.tool_calls:
                    new_content.append(
                        AnthropicResponseContentBlockToolUse(
                            type="tool_use",
                            id=tool_call.id,
                            name=tool_call.function.name or "",
                            input=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                        )
                    )
            elif choice.message.content is not None:
                new_content.append(
                    AnthropicResponseContentBlockText(
                        type="text", text=choice.message.content
                    )
                )

        return new_content

    def _translate_openai_finish_reason_to_anthropic(
        self, openai_finish_reason: str
    ) -> AnthropicFinishReason:
        if openai_finish_reason == "stop":
            return "end_turn"
        elif openai_finish_reason == "length":
            return "max_tokens"
        elif openai_finish_reason == "tool_calls":
            return "tool_use"
        return "end_turn"

    def translate_openai_response_to_anthropic(
        self, response: ModelResponse
    ) -> AnthropicMessagesResponse:
        ## translate content block
        anthropic_content = self._translate_openai_content_to_anthropic(choices=response.choices)  # type: ignore
        ## extract finish reason
        anthropic_finish_reason = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason  # type: ignore
        )
        # extract usage
        usage: Usage = getattr(response, "usage")
        anthropic_usage = AnthropicUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )
        translated_obj = AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=anthropic_content,  # type: ignore
            stop_reason=anthropic_finish_reason,
        )

        return translated_obj

    def _translate_streaming_openai_chunk_to_anthropic_content_block(
        self, choices: List[OpenAIStreamingChoice]
    ) -> Tuple[
        Literal["text", "tool_use"],
        "ContentBlockContentBlockDict",
    ]:
        import uuid

        from litellm.types.llms.anthropic import TextBlock, ToolUseBlock

        for choice in choices:
            if choice.delta.content is not None and len(choice.delta.content) > 0:
                return "text", TextBlock(type="text", text="")
            elif (
                choice.delta.tool_calls is not None
                and len(choice.delta.tool_calls) > 0
                and choice.delta.tool_calls[0].function is not None
            ):
                return "tool_use", ToolUseBlock(
                    type="tool_use",
                    id=choice.delta.tool_calls[0].id or str(uuid.uuid4()),
                    name=choice.delta.tool_calls[0].function.name or "",
                    input={},
                )

        return "text", TextBlock(type="text", text="")

    def _translate_streaming_openai_chunk_to_anthropic(
        self, choices: List[OpenAIStreamingChoice]
    ) -> Tuple[
        Literal["text_delta", "input_json_delta"],
        Union[ContentTextBlockDelta, ContentJsonBlockDelta],
    ]:

        text: str = ""
        partial_json: Optional[str] = None
        for choice in choices:
            if choice.delta.content is not None:
                text += choice.delta.content
            elif choice.delta.tool_calls is not None:
                partial_json = ""
                for tool in choice.delta.tool_calls:
                    if (
                        tool.function is not None
                        and tool.function.arguments is not None
                    ):
                        partial_json += tool.function.arguments

        if partial_json is not None:
            return "input_json_delta", ContentJsonBlockDelta(
                type="input_json_delta", partial_json=partial_json
            )
        else:
            return "text_delta", ContentTextBlockDelta(type="text_delta", text=text)

    def translate_streaming_openai_response_to_anthropic(
        self, response: ModelResponse, current_content_block_index: int
    ) -> Union[ContentBlockDelta, MessageBlockDelta]:
        ## base case - final chunk w/ finish reason
        if response.choices[0].finish_reason is not None:
            delta = MessageDelta(
                stop_reason=self._translate_openai_finish_reason_to_anthropic(
                    response.choices[0].finish_reason
                ),
            )
            if getattr(response, "usage", None) is not None:
                litellm_usage_chunk: Optional[Usage] = response.usage  # type: ignore
            elif (
                hasattr(response, "_hidden_params")
                and "usage" in response._hidden_params
            ):
                litellm_usage_chunk = response._hidden_params["usage"]
            else:
                litellm_usage_chunk = None
            if litellm_usage_chunk is not None:
                usage_delta = UsageDelta(
                    input_tokens=litellm_usage_chunk.prompt_tokens or 0,
                    output_tokens=litellm_usage_chunk.completion_tokens or 0,
                )
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            return MessageBlockDelta(
                type="message_delta", delta=delta, usage=usage_delta
            )
        (
            type_of_content,
            content_block_delta,
        ) = self._translate_streaming_openai_chunk_to_anthropic(
            choices=response.choices  # type: ignore
        )
        return ContentBlockDelta(
            type="content_block_delta",
            index=current_content_block_index,
            delta=content_block_delta,
        )

_TRANS_PARAMS = ["messages", "metadata", "system", "tool_choice", "tools"]

_MSG_ROLE_USER = "user"

_MSG_ROLE_ASSISTANT = "assistant"

_MSG_ROLE_TOOL = "tool"

_TYPE_TEXT = "text"

_TYPE_IMAGE = "image"

_TYPE_TOOL_RESULT = "tool_result"

_TYPE_TOOL_USE = "tool_use"

_TYPE_FUNCTION = "function"

_TYPE_IMAGE_URL = "image_url"

_TOOLS_MAPPED_PARAMS = ("name", "input_schema", "description")
