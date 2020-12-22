package com.pjproductions.rest.definition.json;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ReadMessages<T> {

    @JsonProperty("friend")
    private final T friend;

    @JsonProperty("upperIndex")
    private final long from;

    @JsonProperty("lowerIndex")
    private final long to;

    public ReadMessages(){
        friend = null;
        from = 0;
        to = 0;
    }

    public ReadMessages(T friend, int from, int to) {
        this.friend = friend;
        this.from = from;
        this.to = to;
    }

    public T getFriend() {
        return friend;
    }

    public long getFrom() {
        return from;
    }

    public long getTo() {
        return to;
    }
}
