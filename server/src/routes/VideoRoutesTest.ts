import { Router, Request, Response, RequestHandler } from 'express';
import multer from 'multer';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { promises as fs } from 'fs';

// Configure Multer pour stocker les fichiers uploadés temporairement
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (['.mp4', '.mov', '.avi', '.mkv'].includes(ext)) {
      cb(null, true); // Fichier valide
    } 
  },
});

const router = Router();

// Endpoint pour uploader et compresser une vidéo
const compressVideo: RequestHandler = async (req: Request, res: Response): Promise<void> => {
  if (!req.file) {
    res.status(400).json({ error: 'Aucun fichier vidéo uploadé' });
    return;
  }

  const inputPath = req.file.path;
  const outputPath = path.join(
    path.dirname(inputPath),
    `${path.basename(inputPath, path.extname(inputPath))}_compressed.mp4`
  );

  try {
    await new Promise<void>((resolve, reject) => {
      ffmpeg(inputPath)
        .videoCodec('libx264')
        .outputOptions(['-crf 28', '-preset fast'])
        .output(outputPath)
        .on('end', () => resolve())
        .on('error', (err: Error) => reject(err))
        .run();
    });

    // Envoyer le fichier compressé au client
    res.download(outputPath, path.basename(outputPath), async (err) => {
      if (err) {
        console.error('Erreur lors de l\'envoi du fichier:', err);
        res.status(500).json({ error: 'Erreur lors de l\'envoi du fichier compressé' });
        return;
      }
      // Nettoyer les fichiers temporaires
      await fs.unlink(inputPath).catch(console.error);
      await fs.unlink(outputPath).catch(console.error);
    });
  } catch (error) {
    console.error('Erreur de compression:', error);
    await fs.unlink(inputPath).catch(console.error);
    res.status(500).json({ error: 'Échec de la compression de la vidéo' });
    return;
  }
};

router.post('/compress', upload.single('video'), compressVideo);

export default router;