package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collection;

public class RoomsList {

    @JsonProperty("number-rooms")
    private final int num;

    @JsonProperty("channels")
    private final Collection<RoomView> channels;

    public RoomsList(int num, Collection<RoomView> channels) {
        this.num = num;
        this.channels = channels;
    }

    public int getNum() {
        return num;
    }

    public Collection<RoomView> getChannels() {
        return channels;
    }
}
