package com.pjproductions.rest.definition.json;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.time.ZonedDateTime;

@JsonPropertyOrder({"id", "from", "to", "message", "time"})
public class MessageJSON<T> extends MessageRequest<T> {

    @JsonProperty("id")
    private final int id;

    @JsonProperty("from")
    private final T from;

    @JsonProperty("time")
    private final ZonedDateTime timestamp;

    public MessageJSON(){
        super(null, null);
        id = 0;
        from = null;
        timestamp = null;
    }

    public MessageJSON(int id, T from, T to, String message, ZonedDateTime timestamp) {
        super(to, message);
        this.id = id;
        this.from = from;
        this.timestamp = timestamp;
    }

    public T getFrom() {
        return from;
    }

    public ZonedDateTime getTimestamp() {
        return timestamp;
    }

    public int getId() {
        return id;
    }
}
