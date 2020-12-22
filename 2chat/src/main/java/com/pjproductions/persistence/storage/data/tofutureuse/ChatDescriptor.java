package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collection;

public class ChatDescriptor {

    @JsonProperty("num-USERS")
    private final int numusers;

    @JsonProperty("num-rooms")
    private final int numrooms;

    @JsonProperty("rooms")
    private final Collection<String> rooms;

    public ChatDescriptor(int numusers, int numrooms, Collection<String> rooms) {
        this.numusers = numusers;
        this.numrooms = numrooms;
        this.rooms = rooms;
    }

    public int getNumusers() {
        return numusers;
    }

    public int getNumrooms() {
        return numrooms;
    }

    public Collection<String> getRooms() {
        return rooms;
    }
}
