package com.pmrjproductions.backend.services;

import com.pmrjproductions.backend.repositories.UserRepository;
import com.pmrjproductions.backend.repositories.VerificationTokenRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class VerificationTokenService {
    private final UserRepository userRepository;
    private final VerificationTokenRepository verificationTokenRepository;
    private final MailSenderService sendingMailService;






}
