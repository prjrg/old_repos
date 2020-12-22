package com.pjproductions.persistence.controllers.impl;

import com.pjproductions.persistence.controllers.TokenController;
import com.pjproductions.persistence.storage.data.tofutureuse.Tokens;
import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.exception.PersistenceException;

public class TokenControllerImp implements TokenController {

    private final Tokens tokens;

    public TokenControllerImp(Tokens tokens) {
        this.tokens = tokens;
    }

    public void addToken(User u, Long id){
        tokens.getTokens().get(u.getId()).addToken(id);
    }

    public void removeToken(User u, Long id){
        tokens.getTokens().get(u.getId()).getTokens().remove(id);
    }

    public void validUserToken(User u, Long id){
        if(!tokens.getTokens().get(u.getId()).validToken(id)) throw new PersistenceException(OperationResult.INVALID_CREDENTIALS);
    }
}
