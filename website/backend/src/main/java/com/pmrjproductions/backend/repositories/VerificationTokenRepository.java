package com.pmrjproductions.backend.repositories;

import com.pmrjproductions.backend.model.VerificationToken;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface VerificationTokenRepository extends JpaRepository<VerificationToken, String> {

    List<VerificationToken> findByUserEmail(String email);
    List<VerificationToken> findByToken(String token);
}
