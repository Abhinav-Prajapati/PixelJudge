'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

// Type definitions
interface ImageData {
  id: string | number;
  originalName: string;
  thumbnailUrl: string;
  originalUrl: string;
  width: number;
  height: number;
  fileSize: number;
  mimeType: string;
  createdAt: string;
  title?: string;
  description?: string;
  thumbnailHeight?: number;
  thumbnailWidth?: number;
}

interface ApiResponse {
  success: boolean;
  data: ImageData[];
  message?: string;
}

const thumbScale : number = 0.6;

const PhotoGallery: React.FC = () => {
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  // Replace with your actual API endpoint
  const BASE_URL: string = 'http://localhost:3000';

  useEffect(() => {
    fetchImages();
  }, []);

  const fetchImages = async (): Promise<void> => {
    try {
      setLoading(true);
      setError('');
      
      // Replace this URL with your actual API endpoint
      const response: Response = await fetch(`${BASE_URL}/files/images`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result: ApiResponse = await response.json();

      if (result.success) {
        setImages(result.data);
      } else {
        setError(result.message || 'Failed to fetch images');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError('Error fetching images: ' + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const openModal = (image: ImageData): void => {
    setSelectedImage(image);
  };

  const closeModal = (): void => {
    setSelectedImage(null);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes: string[] = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleModalClick = (e: React.MouseEvent<HTMLDivElement>): void => {
    e.stopPropagation();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-8">
        <div className="text-red-600 mb-4">{error}</div>
        <button
          onClick={fetchImages}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-8">Photo Gallery</h1>

      {images.length === 0 ? (
        <div className="text-center text-gray-500">No images found</div>
      ) : (
        <>
          <div className="text-center mb-6 text-gray-600">
            {images.length} image{images.length !== 1 ? 's' : ''}
          </div>

          {/* Gallery Grid */}
          <div className="flex justify-start border gap-x-2 ">
            {images.map((image: ImageData) => (
              <div
                key={image.id}
                className="relative group cursor-pointer overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-300"
                onClick={() => openModal(image)}
              >
                <div className="relative ">
                  <Image
                    src={image.thumbnailUrl}
                    alt={image.originalName}
                    height={image.thumbnailHeight! * thumbScale}
                    width={image.thumbnailWidth! * thumbScale}
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity duration-300" />
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div
            className="relative max-w-4xl max-h-full bg-white rounded-lg shadow-2xl overflow-hidden"
            onClick={handleModalClick}
          >
            {/* Close button */}
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 z-10 bg-black bg-opacity-50 text-white rounded-full w-10 h-10 flex items-center justify-center hover:bg-opacity-70 transition-colors"
              aria-label="Close modal"
            >
              ×
            </button>

            {/* Image */}
            <div className="relative">
              <Image
                src={selectedImage.originalUrl}
                alt={selectedImage.originalName}
                width={selectedImage.width}
                height={selectedImage.height}
                className="max-w-full max-h-[80vh] object-contain"
              />
            </div>

            {/* Image info */}
            <div className="p-6 bg-white">
              <h3 className="text-xl font-semibold mb-2">{selectedImage.originalName}</h3>
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <span className="font-medium">Dimensions:</span> {selectedImage.width} × {selectedImage.height}
                </div>
                <div>
                  <span className="font-medium">File Size:</span> {formatFileSize(selectedImage.fileSize)}
                </div>
                <div>
                  <span className="font-medium">Type:</span> {selectedImage.mimeType}
                </div>
                <div>
                  <span className="font-medium">Created:</span> {formatDate(selectedImage.createdAt)}
                </div>
              </div>
              {selectedImage.title && (
                <div className="mt-4">
                  <span className="font-medium">Title:</span> {selectedImage.title}
                </div>
              )}
              {selectedImage.description && (
                <div className="mt-2">
                  <span className="font-medium">Description:</span> {selectedImage.description}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PhotoGallery;