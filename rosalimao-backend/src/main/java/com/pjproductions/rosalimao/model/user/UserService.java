package com.pjproductions.rosalimao.model.user;

import com.pjproductions.rosalimao.model.user.item.User;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Transactional
    public void createUser(String email, String firstName, String lastName, String phoneNumber, boolean isAdmin, String passHash) {
        User u = new User(email, firstName, lastName, phoneNumber, passHash, isAdmin);
        userRepository.save(u);
    }

    public long findUserByEmail(String email)  {
        return userRepository.findByEmail(email).getId();
    }

    public User findUser(String email) {
        return userRepository.findByEmail(email);
    }

    @Transactional
    public void updatePassword(String email, String passHash) {
        User u = userRepository.findByEmail(email);
        u.setPassHash(passHash);
        userRepository.save(u);
    }

    @Transactional
    public void updateFirstName(String email, String firstName) {
        User u = userRepository.findByEmail(email);
        u.setFirstName(firstName);
        userRepository.save(u);
    }

    @Transactional
    public void updateLastName(String email, String lastName) {
        User u = userRepository.findByEmail(email);
        u.setLastName(lastName);
        userRepository.save(u);
    }

    @Transactional
    public void updatePhoneNumber(String email, String phoneNumber){
        User u = userRepository.findByEmail(email);
        u.setPhoneNumber(phoneNumber);
        userRepository.save(u);
    }

    @Transactional
    public void updateEmail(String email, String newEmail){
        User u = userRepository.findByEmail(email);
        u.setEmail(newEmail);
        userRepository.save(u);
    }
}
