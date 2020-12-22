package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.ZonedDateTime;
import java.util.Collection;

public class ChannelMessage {

    @JsonProperty("sender")
    private final Long sender;
    @JsonProperty("receivers")
    private final Collection<Long> receivers;
    @JsonProperty("message")
    private final String message;

    @JsonFormat(shape=JsonFormat.Shape.STRING, pattern = "dd-MM-yyyy hh:mm:ss")
    @JsonProperty("time")
    private final ZonedDateTime timestamp;

    public ChannelMessage(Long sender, Collection<Long> receivers, String message, ZonedDateTime timestamp) {
        this.sender = sender;
        this.receivers = receivers;
        this.message = message;
        this.timestamp = timestamp;
    }

    public Long getSender() {
        return sender;
    }

    public Collection<Long> getReceivers() {
        return receivers;
    }

    public String getMessage() {
        return message;
    }

    public ZonedDateTime getTimestamp() {
        return timestamp;
    }
}
