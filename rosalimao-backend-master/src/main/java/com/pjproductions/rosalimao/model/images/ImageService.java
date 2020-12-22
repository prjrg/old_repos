package com.pjproductions.rosalimao.model.images;

import com.pjproductions.rosalimao.model.images.item.Image;
import com.pjproductions.rosalimao.model.images.item.Model;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.transaction.Transactional;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

@Service
public class ImageService {

    private final ImageModelRepository repository;
    private final PictureRepository pictureRepository;

    public ImageService(ImageModelRepository repository, PictureRepository pictureRepository) {
        this.repository = repository;
        this.pictureRepository = pictureRepository;
    }

    @Transactional
    public long saveModel(String title, String description) {
        Model model = repository.save(new Model(title, description));

        return model.getId();
    }

    @Transactional
    public void saveImageFile(Long id, MultipartFile file) throws IOException {


        if(file.isEmpty()) throw new IOException("File " + file + " must have size>0");

        Optional<Model> modelOption = repository.findById(id);

        Model model = modelOption.orElseThrow();

        Image im = new Image(file.getOriginalFilename(), file.getContentType(), file.getBytes(), model);
        model.getImages().add(im);
        repository.save(model);
        pictureRepository.save(im);
    }

    public List<Long> getAllModelIds() {
        return StreamSupport.stream(repository.findAll().spliterator(), false).map(Model::getId).collect(Collectors.toList());
    }

    public List<Long> getAllImagesIds(long id) {
        Model model = repository.findById(id).orElseThrow();

        return repository.queryByModel(model);
    }

    public List<Model> getAllModels(){
        return StreamSupport.stream(repository.findAll().spliterator(), false).collect(Collectors.toList());
    }

    public Model getImageModel(Long id){
        return repository.findById(id).orElse(null);
    }

    public byte[] getImage(long idPicture) {
        return pictureRepository.findById(idPicture).orElseThrow().getPicture();
    }

}
