package com.pjproductions.persistence.storage.data.tofutureuse;


import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Tokens {

    @JsonProperty("user-TOKENS")
    private final Map<Long, UserTokens> tokens;

    public Tokens(Map<Long, UserTokens> tokens) {
        this.tokens = tokens;
    }

    public Tokens() {
        tokens = new ConcurrentHashMap<>();
    }

    public Map<Long, UserTokens> getTokens() {
        return tokens;
    }

}
