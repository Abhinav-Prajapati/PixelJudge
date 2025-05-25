/*
TODO: add toster for file upload status []
TODO: add support to upload multiple files []
TODO: add support to display if image is dublicate or not (dublicate detectionis implemented in api) []
*/

import React, { useState, useEffect, useRef } from 'react';
import { Upload, X, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'react-toastify';



interface UploadedFile {
  name: string;
  size: number;
  type: string;
}

type UploadStatus = 'success' | 'error' | null;

const FileUploader: React.FC = () => {
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const dragCounter = useRef<number>(0);

  useEffect(() => {
    const handleDragEnter = (e: DragEvent): void => {
      e.preventDefault();
      dragCounter.current++;
      if (e.dataTransfer?.items && e.dataTransfer.items.length > 0) {
        setIsDragOver(true);
      }
    };

    const handleDragLeave = (e: DragEvent): void => {
      e.preventDefault();
      dragCounter.current--;
      if (dragCounter.current === 0) {
        setIsDragOver(false);
      }
    };

    const handleDragOver = (e: DragEvent): void => {
      e.preventDefault();
    };

    const handleDrop = (e: DragEvent): void => {
      e.preventDefault();
      setIsDragOver(false);
      dragCounter.current = 0;
      
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    };

    document.addEventListener('dragenter', handleDragEnter);
    document.addEventListener('dragleave', handleDragLeave);
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('drop', handleDrop);

    return () => {
      document.removeEventListener('dragenter', handleDragEnter);
      document.removeEventListener('dragleave', handleDragLeave);
      document.removeEventListener('dragover', handleDragOver);
      document.removeEventListener('drop', handleDrop);
    };
  }, []);

  const handleFileUpload = async (file: File): Promise<void> => {
    setIsUploading(true);
    setUploadStatus(null);
    setErrorMessage('');
    
    const uploadedFileInfo: UploadedFile = {
      name: file.name,
      size: file.size,
      type: file.type
    };
    setUploadedFile(uploadedFileInfo);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:3000/files/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.status === 201) {
        setUploadStatus('success');
      } else {
        const errorText = await response.text();
        throw new Error(errorText || `Upload failed with status ${response.status}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMsg = error instanceof Error ? error.message : 'An unexpected error occurred';
      setErrorMessage(errorMsg);
      setUploadStatus('error');
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = (): void => {
    setUploadStatus(null);
    setUploadedFile(null);
    setIsUploading(false);
    setErrorMessage('');
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <>
      {/* Drag overlay */}
      {isDragOver && (
        <div className="fixed inset-0 z-50 bg-blue-500 bg-opacity-20 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-white rounded-2xl shadow-2xl p-12 border-2 border-dashed border-blue-400 max-w-md mx-4">
            <div className="text-center">
              <Upload className="mx-auto h-16 w-16 text-blue-500 mb-4" />
              <h3 className="text-2xl font-semibold text-gray-900 mb-2">
                Drop your file here
              </h3>
              <p className="text-gray-600">
                Release to upload your file
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Upload status modal */}
      {(isUploading || uploadStatus) && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4 relative">
            {!isUploading && (
              <button
                onClick={resetUpload}
                className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="h-5 w-5 text-gray-500" />
              </button>
            )}

            <div className="text-center">
              {/* Uploading state */}
              {isUploading && (
                <>
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Uploading...
                  </h3>
                  {uploadedFile && (
                    <div className="text-left bg-gray-50 rounded-lg p-4 mb-4">
                      <p className="font-medium text-gray-900 truncate">
                        {uploadedFile.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {formatFileSize(uploadedFile.size)}
                      </p>
                    </div>
                  )}
                  <p className="text-gray-600">
                    Please wait while we upload your file...
                  </p>
                </>
              )}

              {/* Success state */}
              {uploadStatus === 'success' && (
                <>
                  <CheckCircle className="mx-auto h-16 w-16 text-green-500 mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Upload Successful!
                  </h3>
                  {uploadedFile && (
                    <div className="text-left bg-gray-50 rounded-lg p-4 mb-4">
                      <p className="font-medium text-gray-900 truncate">
                        {uploadedFile.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {formatFileSize(uploadedFile.size)}
                      </p>
                    </div>
                  )}
                  <p className="text-gray-600 mb-6">
                    Your file has been uploaded successfully.
                  </p>
                  <button
                    onClick={resetUpload}
                    className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                  >
                    Done
                  </button>
                </>
              )}

              {/* Error state */}
              {uploadStatus === 'error' && (
                <>
                  <AlertCircle className="mx-auto h-16 w-16 text-red-500 mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Upload Failed
                  </h3>
                  {uploadedFile && (
                    <div className="text-left bg-gray-50 rounded-lg p-4 mb-4">
                      <p className="font-medium text-gray-900 truncate">
                        {uploadedFile.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {formatFileSize(uploadedFile.size)}
                      </p>
                    </div>
                  )}
                  <div className="text-left bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                    <p className="text-sm text-red-700">
                      <strong>Error:</strong> {errorMessage || 'Upload failed. Please try again.'}
                    </p>
                  </div>
                  <div className="flex space-x-3">
                    <button
                      onClick={() => uploadedFile && handleFileUpload(new File([], uploadedFile.name, { type: uploadedFile.type }))}
                      className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                    >
                      Retry
                    </button>
                    <button
                      onClick={resetUpload}
                      className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FileUploader;