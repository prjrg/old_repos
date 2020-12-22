package com.pjproductions.persistence.storage.data;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.concurrent.ConcurrentSkipListMap;

public class Messages<T> {

    @JsonProperty("messagesById")
    protected final ConcurrentSkipListMap<Long, T> messagesById;

    @JsonProperty("index-counter")
    protected final IdGenerator idGenerator;

    public Messages(ConcurrentSkipListMap<Long, T> messagesById,  long indexCounter) {
        this.messagesById = messagesById;
        this.idGenerator = IdGenerator.defaultGenerator(indexCounter);
    }

    public Messages(){
        this(new ConcurrentSkipListMap<>(), (long) 0);
    }

    public static<T> Messages<T> create(ConcurrentSkipListMap<Long, T> messages, long indexCounter){
        return new Messages<>(messages,indexCounter);
    }

}
