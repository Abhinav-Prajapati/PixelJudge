export class CreateFileDto {
  originalName: string;
  filename: string;
  filePath: string;
  checksum: string;
  mimeType?: string;
  fileSize?: number;
  width?: number;
  height?: number;
  title?: string;
  description?: string;
  tags?: string[];
  thumbnailFilename?: string;
  thumbnailFilePath?: string;
}
