package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Set;

public class RoomView {

    @JsonProperty("name")
    private final String name;

    @JsonProperty("active-USERS")
    private final Set<Long> activeUsersId;

    @JsonProperty("number-from-USERS")
    private final int numUsers;

    public RoomView(String name, Set<Long> activeUsersId, int numUsers) {
        this.name = name;
        this.activeUsersId = activeUsersId;
        this.numUsers = numUsers;
    }

    public String getName() {
        return name;
    }

    public Set<Long> getActiveUsersId() {
        return activeUsersId;
    }

    public int getNumUsers() {
        return numUsers;
    }

    public static RoomView fromRoom(Room r){
        return new RoomView(r.getName(), r.getActiveUsers(), r.getActiveUsers().size());
    }
}
