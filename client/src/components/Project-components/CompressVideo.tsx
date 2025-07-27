import React, { useState } from 'react';
import axios from 'axios';

const VideoCompressor: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setMessage('');
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) {
      setMessage('Veuillez sélectionner une vidéo.');
      return;
    }

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post('http://localhost:8080/vid/compress', formData, {
        responseType: 'blob', // Important pour recevoir le fichier compressé
      });

      // Créer un lien pour télécharger le fichier compressé
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${file.name.split('.')[0]}_compressed.mp4`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setMessage('Vidéo compressée avec succès !');
    } catch (error) {
      setMessage('Erreur lors de la compression de la vidéo.');
      console.error(error);
    }
  };

  return (
    <div className="max-w-md mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Compresseur de Vidéo</h1>
      <p className="mb-4">Choisissez une vidéo à compresser :</p>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".mp4,.mov,.avi,.mkv"
          onChange={handleFileChange}
          className="mb-4 p-2 border rounded w-full"
        />
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          disabled={!file}
        >
          Compresser
        </button>
      </form>
      {message && <p className="mt-4 text-red-500">{message}</p>}
    </div>
  );
};

export default VideoCompressor;