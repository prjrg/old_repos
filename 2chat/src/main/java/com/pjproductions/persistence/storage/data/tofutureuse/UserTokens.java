package com.pjproductions.persistence.storage.data.tofutureuse;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class UserTokens {

    @JsonProperty("valid-TOKENS")
    private final ConcurrentMap<Long, Boolean> tokens;

    public UserTokens(ConcurrentMap<Long, Boolean> tokens) {
        this.tokens = tokens;
    }

    public UserTokens(){
        this.tokens = new ConcurrentHashMap<>();
    }

    public ConcurrentMap<Long, Boolean> getTokens() {
        return tokens;
    }

    public void addToken(Long id){
        tokens.put(id, true);
    }

    public boolean validToken(Long id){
        return tokens.get(id) != null;
    }
}
