package com.pjproductions.rest.security.validation.validators;

import com.pjproductions.rest.exception.ChatException;

import java.util.Optional;

public interface CheckValid<T, U extends ChatException> {

    Optional<U> isValid(T t);
}
