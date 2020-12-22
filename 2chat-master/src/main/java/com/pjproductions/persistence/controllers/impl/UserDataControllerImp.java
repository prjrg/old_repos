package com.pjproductions.persistence.controllers.impl;

import com.pjproductions.persistence.controllers.UserDataController;
import com.pjproductions.persistence.storage.Storage;
import com.pjproductions.persistence.storage.data.Affinity;
import com.pjproductions.persistence.storage.data.User;
import com.pjproductions.rest.definition.OperationResult;
import com.pjproductions.rest.definition.json.Friend;
import com.pjproductions.rest.exception.ChatException;
import com.pjproductions.rest.exception.PersistenceException;
import com.pjproductions.rest.exception.request.AccountExceptionChat;
import com.pjproductions.rest.exception.request.EmailExceptionChat;

import java.util.Collection;
import java.util.Collections;
import java.util.function.Function;

public class UserDataControllerImp implements UserDataController {
    private final Storage storage;

    public UserDataControllerImp(Storage storage) {
        this.storage = storage;
    }

    @Override
    public <U extends ChatException> User findUserByEmailOrName(String identifier) throws U {
        return storage.USER_OPERATIONS.findUserByEmailOrName(identifier);
    }

    public void addUser(String username, String email, String passHash) throws EmailExceptionChat, AccountExceptionChat {
        storage.USER_OPERATIONS.addUser(username, email, passHash);
    }


    public boolean isFriend(User u, String friend) {
        User u0 = findUserByEmailOrName(friend);

        return storage.USER_OPERATIONS.isFriend(u, u0);
    }

    @Override
    public <U extends ChatException> void removeFriend(String user, String friend) throws U {
        User u = findUserByEmailOrName(user);
        User u0 = findUserByEmailOrName(friend);

        storage.USER_OPERATIONS.removeFriend(u, u0);
    }

    public void blockUser(String username, String friend) {
        User u = findUserByEmailOrName(username);
        User u0 = findUserByEmailOrName(friend);

        if(u.getId().equals(u0.getId())) throw new PersistenceException(OperationResult.INVALID_OPERATION);

        storage.USER_OPERATIONS.blockUser(u, u0);
    }

    public boolean isBlocked(User user, String friend){
        User u0 = findUserByEmailOrName(friend);

        return storage.USER_OPERATIONS.isBlocked(user, u0);
    }

    @Override
    public <U extends ChatException> User findUserById(Long id) throws U {
        return storage.USER_OPERATIONS.findUserById(id);
    }

    public <U extends ChatException> Collection<Friend> userFriends(String username) throws U{
        User u = findUserByEmailOrName(username);
        Collection<Friend> friends = storage.USER_OPERATIONS.getAffinities(u).mapFriend(MAP_AFFINITY_TO_FRIEND);
        Collection<Friend> friendsRequest = storage.USER_OPERATIONS.getAffinities(u).mapSentRequests(MAP_ID_TO_FRIEND);
        friends.addAll(friendsRequest);
        return Collections.unmodifiableCollection(friends);
    }

    @Override
    public <U extends ChatException> void sendRequest(String user, String friend) throws U {
        User u = findUserByEmailOrName(user);
        User f = findUserByEmailOrName(friend);

        if(u.getId().equals(f.getId())) throw new ChatException(OperationResult.INVALID_OPERATION);

        storage.MESSAGES_OPERATIONS.createMessagesForTwoUsers(u, f);
        storage.USER_OPERATIONS.getAffinities(u).addSentRequest(f);
        storage.USER_OPERATIONS.getAffinities(f).addFriendRequest(u);
    }

    @Override
    public <U extends ChatException> void handleFriendRequest(String username, String friend, boolean accept) throws U {
        User u = findUserByEmailOrName(username);
        User uf = findUserByEmailOrName(friend);

        if(u.getId().equals(uf.getId())) throw new ChatException(OperationResult.INVALID_OPERATION);

        storage.USER_OPERATIONS.getAffinities(u).promoteRequestToFriend(uf, accept);
        if(accept){
            storage.MESSAGES_OPERATIONS.createMessagesForTwoUsers(u, uf);
            storage.USER_OPERATIONS.getAffinities(uf).requestAccepted(u);
        }
    }


    @Override
    public <U extends ChatException> Collection<String> friendRequests(String username) throws U {
        User u = findUserByEmailOrName(username);
        return storage.USER_OPERATIONS.getAffinities(u).mapFriendRequests((id) -> findUserById(id).getUsername());
    }

    @Override
    public <U extends ChatException> User isPasswordCorrect(String username, String passHash) {
        User u = findUserByEmailOrName(username);

        return new User(u.getId(), u.getUsername(), u.getEmail(), passHash.equals(u.getPasswordHash()) ? "" : "NO");
    }

    private final Function<Affinity, Friend> MAP_AFFINITY_TO_FRIEND = new Function<Affinity, Friend>() {
        @Override
        public Friend apply(Affinity affinity) {
            long id = affinity.getUserId();

            String username = storage.USER_OPERATIONS.findUserById(id).getUsername();

            return Friend.of(username, affinity.isBlocked());
        }
    };

    private final Function<Long, Friend> MAP_ID_TO_FRIEND = new Function<Long, Friend>() {
        @Override
        public Friend apply(Long aLong) {
            String username = storage.USER_OPERATIONS.findUserById(aLong).getUsername();

            return Friend.of(username);
        }
    };


}
