export class CreateFileDto {
  originalName: string;
  filename: string;
  filePath: string;
  checksum: string;
  mimeType: string;
  fileSize: number;
  width?: number | null;
  height?: number | null;
  title?: string | null;
  description?: string | null;
  tags?: string[] | null;
}
