from openai.types import realtime

session_events = {
    "session.created": realtime.SessionCreatedEvent,
    "session.updated": realtime.SessionUpdatedEvent,
}
input_audio_buffer_events = {
    "input_audio_buffer.speech_started": realtime.InputAudioBufferSpeechStartedEvent,
    "input_audio_buffer.speech_stopped": realtime.InputAudioBufferSpeechStoppedEvent,
    "input_audio_buffer.committed": realtime.InputAudioBufferCommittedEvent,
}
conversation_item_events = {
    "conversation.item.added": realtime.ConversationItemAdded,
    "conversation.item.done": realtime.ConversationItemDone,
}
response_events = {
    "response.created": realtime.ResponseCreatedEvent,
    "response.output_item.added": realtime.ResponseOutputItemAddedEvent,
    "response.content_part.added": realtime.ResponseContentPartAddedEvent,
    "response.text.delta": realtime.ResponseTextDeltaEvent,
    "response.function_call_arguments.delta": realtime.ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": realtime.ResponseFunctionCallArgumentsDoneEvent,
    "response.text.done": realtime.ResponseTextDoneEvent,
    "response.content_part.done": realtime.ResponseContentPartDoneEvent,
    "response.output_item.done": realtime.ResponseOutputItemDoneEvent,
    "response.done": realtime.ResponseDoneEvent,
}
rate_limits_events = {
    "rate_limits.updated": realtime.RateLimitsUpdatedEvent,
}


event_dict = (
    session_events
    | input_audio_buffer_events
    | conversation_item_events
    | response_events
    | rate_limits_events
)
