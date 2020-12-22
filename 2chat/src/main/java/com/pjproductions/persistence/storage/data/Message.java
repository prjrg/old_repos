package com.pjproductions.persistence.storage.data;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.ZonedDateTime;

@JsonAutoDetect(
        fieldVisibility = JsonAutoDetect.Visibility.DEFAULT,
        setterVisibility = JsonAutoDetect.Visibility.NONE,
        getterVisibility = JsonAutoDetect.Visibility.NONE,
        isGetterVisibility = JsonAutoDetect.Visibility.NONE,
        creatorVisibility = JsonAutoDetect.Visibility.NONE
)
public class Message {
    @JsonProperty("order-id")
    private Long id = Long.MAX_VALUE;

    @JsonProperty("sender")
    private final Long sender;
    @JsonProperty("receiver")
    private final Long receiver;
    @JsonProperty("message")
    private final String message;

    @JsonProperty("has-read")
    private boolean isRead;

    @JsonProperty("time")
    private final ZonedDateTime timestamp;

    public Message(Long sender, Long receiver, String message, ZonedDateTime timestamp, boolean isRead) {
        this.sender = sender;
        this.receiver = receiver;
        this.message = message;
        this.timestamp = timestamp;
        this.isRead = isRead;
    }

    public Long getSender() {
        return sender;
    }

    public Long getReceiver() {
        return receiver;
    }

    public String getMessage() {
        return message;
    }

    public ZonedDateTime getTimestamp() {
        return timestamp;
    }

    public boolean isRead() {
        return isRead;
    }

    public void setRead(boolean read) {
        isRead = read;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }
}
