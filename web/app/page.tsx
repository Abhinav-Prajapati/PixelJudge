'use client';
import FileUploader from '@/components/FileUpLoader';
import Image from 'next/image';
import { useEffect, useState } from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

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

const thumbScale: number = 0.6;

const PhotoGallery: React.FC = () => {
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchImages();
  }, []);

  const fetchImages = async (): Promise<void> => {
    try {
      setLoading(true);
      setError('');

      // Replace this URL with your actual API endpoint
      const response: Response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/files/images`);

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
          {/* Gallery Grid Grouped by Date */}
          <div className="space-y-8">
            {Object.entries(images).map(([date, dateImages]) => (
              <div key={date} className="space-y-4">
                {/* Date Header */}
                <div className="flex items-center space-x-4">
                  <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
                    {new Date(date).toLocaleDateString('en-US', {
                      weekday: 'long',
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </h2>
                  <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600"></div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {dateImages.length} {dateImages.length === 1 ? 'image' : 'images'}
                  </span>
                </div>

                {/* Images Grid for this Date */}
                <div className="flex flex-wrap justify-start gap-x-2 gap-y-6">
                  {dateImages.map((image: ImageData) => {
                    const isLandscape = image.thumbnailWidth! > image.thumbnailHeight!;
                    const aspectRatio = 16 / 11; // looks good to eyes then 16 / 9
                    const baseWidth = 100;
                    const containerWidth = isLandscape ? (baseWidth * aspectRatio * aspectRatio) : baseWidth;
                    const containerHeight = isLandscape ? (baseWidth * aspectRatio) : baseWidth * aspectRatio;

                    return (
                      <div
                        key={image.id}
                        className="relative group cursor-pointer overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-300"
                        style={{
                          width: `${containerWidth}px`,
                          height: `${containerHeight}px`
                        }}
                        onClick={() => openModal(image)}
                      >
                        <Image
                          src={image.thumbnailUrl}
                          alt={image.originalName}
                          fill
                          className="object-cover"
                        />
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity duration-300" />
                      </div>
                    );
                  })}
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
      <FileUploader />
      <ToastContainer aria-label="Notification messages" />
    </div>
  );
};

export default PhotoGallery;
