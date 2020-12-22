package com.pjproductions.rest.definition.json;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonPropertyOrder({"to", "message"})
public class MessageRequest<T> {

    @JsonProperty("to")
    private final T to;

    @JsonProperty("message")
    private final String message;

    public MessageRequest(){
        to = null;
        message = "";
    }

    public MessageRequest(T to, String message) {
        this.to = to;
        this.message = message;
    }

    public T to() {
        return to;
    }

    public String message() {
        return message;
    }
}
