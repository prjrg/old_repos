package com.pjproductions.rosalimao.controller;

import com.pjproductions.rosalimao.model.images.ImageService;
import com.pjproductions.rosalimao.model.images.item.Model;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Controller
@RequestMapping(path = "/image")
public class ImageController {

    private final ImageService imageService;

    @Autowired
    public ImageController(ImageService imageService){
        this.imageService = imageService;
    }

    @PostMapping("/upload")
    public ResponseEntity<Long> uploadModel(@RequestParam("title") String title, @RequestParam("description") String description) throws IOException {
        long res = imageService.saveModel(title, description);

        return ResponseEntity.status(HttpStatus.CREATED).body(res);
    }

    @PostMapping("/uploadFile")
    public ResponseEntity<Long> uploadImage(@RequestParam("id") Long id, @RequestParam("file") MultipartFile file) throws IOException {
        imageService.saveImageFile(Long.valueOf(id), file);

        return ResponseEntity.status(HttpStatus.CREATED).body(Long.MAX_VALUE);
    }

    @GetMapping(path = "/ids")
    public ResponseEntity<List<Long>> getIds(){
        return ResponseEntity.status(HttpStatus.OK).body(imageService.getAllModelIds());
    }

    @GetMapping(path = "/ids/{modelId}")
    public ResponseEntity<List<Long>> getPictureIds(@PathVariable("modelId") Long id){
        return ResponseEntity.status(HttpStatus.OK).body(imageService.getAllImagesIds(id));
    }

    @GetMapping(path = "/getImage/{imageId}", produces = {MediaType.IMAGE_JPEG_VALUE, MediaType.IMAGE_PNG_VALUE})
    public @ResponseBody byte[] getImage(@PathVariable("imageId") Long id) {
        return imageService.getImage(id);
    }

    @GetMapping(path = "/getElement/{imageId}")
    public ResponseEntity<Model> getImageModel(@PathVariable("imageId") Long id){
        return ResponseEntity.ok(imageService.getImageModel(id));
    }

}
