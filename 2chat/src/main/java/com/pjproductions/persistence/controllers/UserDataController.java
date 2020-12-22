package com.pjproductions.persistence.controllers;

import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.rest.definition.json.Friend;
import com.pjproductions.rest.exception.ChatException;

import java.util.Collection;

public interface UserDataController {

    <U extends ChatException> User findUserByEmailOrName(String identifier) throws U;
    <U extends ChatException> User findUserById(Long id) throws U;

    <U extends ChatException> void addUser(String username, String email, String passHash) throws U;
    boolean isFriend(User u, String friend);

    <U extends ChatException> void removeFriend(String u, String friend) throws U;

    <U extends ChatException> void blockUser(String u, String user) throws U;
    boolean isBlocked(User u, String user);


    <U extends ChatException> Collection<Friend> userFriends(String username) throws U;

    <U extends ChatException> void sendRequest(String user, String friend) throws U;

    <U extends ChatException> void handleFriendRequest(String username, String friend, boolean accept) throws U;

    <U extends ChatException> Collection<String> friendRequests(String username) throws U;

    <U extends ChatException> User isPasswordCorrect(String username, String passHash) throws U;


}
