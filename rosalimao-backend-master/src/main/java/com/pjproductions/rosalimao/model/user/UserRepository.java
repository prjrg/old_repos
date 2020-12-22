package com.pjproductions.rosalimao.model.user;

import com.pjproductions.rosalimao.model.user.item.User;
import org.springframework.data.repository.PagingAndSortingRepository;

public interface UserRepository extends PagingAndSortingRepository<User, Long> {

    public User findByEmail(String email);
}
