package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class Room {

    @JsonProperty("name")
    private final String name;

    @JsonProperty("active-USERS")
    private final Set<Long> activeUsers;

    @JsonProperty("creator")
    private final Long creator;

    @JsonProperty("master-room")
    private final boolean master;

    public Room(String name, Long creator, boolean master) {
        this.name = name;
        ConcurrentHashMap<Long, Long> map = new ConcurrentHashMap<>();
        this.activeUsers = map.newKeySet();
        this.creator = creator;
        this.master = master;
    }

    public String getName() {
        return name;
    }

    public Set<Long> getActiveUsers() {
        return activeUsers;
    }

    public void addUser(Long userId){
        activeUsers.add(userId);
    }

    public Long getCreator() {
        return creator;
    }

    public boolean isMaster() {
        return master;
    }
}
