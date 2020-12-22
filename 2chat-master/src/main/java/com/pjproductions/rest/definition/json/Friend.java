package com.pjproductions.rest.definition.json;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Friend {

    @JsonProperty("username")
    private final String username;

    @JsonProperty("blocked")
    private final boolean blocked;

    @JsonProperty("accepted")
    private final boolean hasAccepted;

    public Friend() {
        username="";
        blocked=false;
        hasAccepted = true;
    }

    public Friend(String username, boolean blocked, boolean hasAccepted) {
        this.username = username;
        this.blocked = blocked;
        this.hasAccepted = hasAccepted;
    }

    public String getUsername() {
        return username;
    }

    public boolean isBlocked() {
        return blocked;
    }

    public static Friend of(String username, boolean blocked){
        return new Friend(username, blocked, true);
    }

    public static Friend of(String username){
        return new Friend(username, false, false);
    }
}
