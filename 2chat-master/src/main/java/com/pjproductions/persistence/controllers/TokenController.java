package com.pjproductions.persistence.controllers;

import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.rest.exception.ChatException;

public interface TokenController {

    <U extends ChatException> void addToken(User u, Long id) throws U;

    void removeToken(User u, Long id);

    <U extends ChatException> void validUserToken(User u, Long id) throws U;
}
