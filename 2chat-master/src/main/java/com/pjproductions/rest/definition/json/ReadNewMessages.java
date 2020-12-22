package com.pjproductions.rest.definition.json;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ReadNewMessages<T> {

    @JsonProperty("friend")
    private final T friend;

    @JsonProperty("lowerIndex")
    private final long to;

    public ReadNewMessages(){
        friend = null;
        to = -1;
    }

    public ReadNewMessages(T friend, long to) {
        this.friend = friend;
        this.to = to;
    }

    public T friend() {
        return friend;
    }

    public long fromIndex() {
        return to;
    }
}
