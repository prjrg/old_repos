package com.pjproductions.persistence.storage.data;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonPropertyOrder({"id","is-friend"})
public class Affinity {

    @JsonProperty("id")
    private final Long userId;

    @JsonProperty("is-blocked")
    private final boolean blocked;

    public Affinity(Long userId, boolean blocked) {
        this.userId = userId;
        this.blocked = blocked;
    }

    public Long getUserId() {
        return userId;
    }

    public boolean isBlocked() {
        return blocked;
    }


}
