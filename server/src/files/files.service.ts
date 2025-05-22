import { Injectable, ConflictException, InternalServerErrorException } from '@nestjs/common';
import { CreateFileDto } from './dto/create-file.dto';
import { PrismaService } from 'prisma/prisma.service';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import * as sharp from 'sharp';

const writeFile = promisify(fs.writeFile);
const unlink = promisify(fs.unlink);

export interface ThumbnailOptions {
  width?: number;
  height?: number;
  quality?: number;
}

@Injectable()
export class FilesService {
  constructor(private prisma: PrismaService) { }

  // Default thumbnail configuration
  private readonly defaultThumbnailOptions: ThumbnailOptions = {
    width: 300,
    height: 300,
    quality: 80,
  };

  /**
   * Creates a new image record in the database
   */
  async createImageRecord(createFileDto: CreateFileDto) {
    try {
      const image = await this.prisma.image.create({
        data: {
          originalName: createFileDto.originalName,
          filename: createFileDto.filename,
          filePath: createFileDto.filePath,
          checksum: createFileDto.checksum,
          mimeType: createFileDto.mimeType,
          fileSize: createFileDto.fileSize,
          width: createFileDto.width,
          height: createFileDto.height,
          title: createFileDto.title,
          description: createFileDto.description,
          tags: createFileDto.tags || undefined,
          thubmnailFilename: createFileDto.thumbnailFilename,
          thumbnailFilePath: createFileDto.thumbnailFilePath,
        },
      });
      return image;
    } catch (error) {
      throw new InternalServerErrorException('Failed to create image record');
    }
  }

  /**
   * Checks if an image with the given checksum already exists
   */
  async doesImageExist(checksum: string): Promise<boolean> {
    try {
      const image = await this.prisma.image.findUnique({
        where: { checksum },
      });
      return !!image;
    } catch (error) {
      throw new InternalServerErrorException('Failed to check image existence');
    }
  }

  /**
   * Finds an existing image by checksum
   */
  async findImageByChecksum(checksum: string) {
    try {
      return await this.prisma.image.findUnique({
        where: { checksum },
      });
    } catch (error) {
      throw new InternalServerErrorException('Failed to find image');
    }
  }

  /**
   * Generates MD5 checksum for file buffer
   */
  generateChecksum(buffer: Buffer): string {
    return crypto.createHash('md5').update(buffer).digest('hex');
  }

  /**
   * Converts image buffer to thumbnail
   * @param imageBuffer - Original image buffer
   * @param options - Thumbnail generation options
   * @returns Promise<Buffer> - Thumbnail buffer
   */

  async convertToThumbnail(
    imageBuffer: Buffer,
    options: ThumbnailOptions = {}
  ): Promise<Buffer> {
    try {
      const { quality = 80 } = { ...this.defaultThumbnailOptions, ...options };

      const thumbnail = await sharp(imageBuffer)
        .resize({ width: 400 }) // Only set width, auto height to maintain aspect ratio
        .jpeg({ quality })      // JPEG format with 70% quality (i.e., 30% downgrade)
        .toBuffer();

      return thumbnail;
    } catch (error) {
      console.error('Thumbnail generation error:', error);
      throw new InternalServerErrorException('Failed to generate thumbnail');
    }
  }
  /**
   * Extracts image metadata using Sharp
   */
  async extractImageMetadata(buffer: Buffer): Promise<{ width: number; height: number }> {
    try {
      const metadata = await sharp(buffer).metadata();
      return {
        width: metadata.width || 0,
        height: metadata.height || 0,
      };
    } catch (error) {
      console.error('Failed to extract image metadata:', error);
      return { width: 0, height: 0 };
    }
  }

  /**
   * Saves file to filesystem with unique filename
   */
  async saveFileToSystem(
    file: Express.Multer.File,
    uploadPath: string
  ): Promise<{ filename: string; filePath: string }> {
    try {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
      const ext = path.extname(file.originalname);
      const filename = `${file.fieldname}-${uniqueSuffix}${ext}`;
      const filePath = path.join(`${uploadPath}/uploads/`, filename);

      await writeFile(filePath, file.buffer);

      return {
        filename,
        filePath: `/uploads/${filename}`,
      };
    } catch (error) {
      console.error('File write error:', error);
      throw new InternalServerErrorException('Failed to save file to filesystem');
    }
  }

  /**
   * Saves thumbnail buffer to filesystem
   */
  async saveThumbnailToSystem(
    thumbnailBuffer: Buffer,
    originalFilename: string,
    uploadPath: string
  ): Promise<{ thumbnailFilename: string; thumbnailFilePath: string }> {
    try {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
      const baseName = path.parse(originalFilename).name;
      const thumbnailFilename = `thumb-${baseName}-${uniqueSuffix}.jpg`;
      const thumbnailPath = path.join(uploadPath, 'thumbnails');

      // Ensure thumbnails directory exists
      if (!fs.existsSync(thumbnailPath)) {
        fs.mkdirSync(thumbnailPath, { recursive: true });
      }

      const fullThumbnailPath = path.join(thumbnailPath, thumbnailFilename);
      await writeFile(fullThumbnailPath, thumbnailBuffer);

      return {
        thumbnailFilename,
        thumbnailFilePath: `/thumbnails/${thumbnailFilename}`,
      };
    } catch (error) {
      console.error('Thumbnail save error:', error);
      throw new InternalServerErrorException('Failed to save thumbnail to filesystem');
    }
  }

  /**
   * Processes image upload with duplicate checking and thumbnail generation
   */
  async processImageUpload(
    file: Express.Multer.File,
    uploadPath: string,
    thumbnailOptions?: ThumbnailOptions
  ): Promise<{
    message: string;
    filename: string;
    path: string;
    thumbnailPath?: string;
    isExisting?: boolean
  }> {
    try {
      // Generate checksum
      const checksum = this.generateChecksum(file.buffer);

      // Check if image already exists
      const existingImage = await this.findImageByChecksum(checksum);
      if (existingImage) {
        return {
          message: 'Image already exists',
          filename: existingImage.filename,
          path: existingImage.filePath,
          thumbnailPath: existingImage.thumbnailFilePath || undefined,
          isExisting: true,
        };
      }

      // Extract image metadata
      const { width, height } = await this.extractImageMetadata(file.buffer);

      // Save original file
      const { filename, filePath } = await this.saveFileToSystem(file, uploadPath);

      // Generate and save thumbnail
      const thumbnailBuffer = await this.convertToThumbnail(file.buffer, thumbnailOptions);
      const { thumbnailFilename, thumbnailFilePath } = await this.saveThumbnailToSystem(
        thumbnailBuffer,
        filename,
        uploadPath
      );

      // Create image record
      const createFileDto: CreateFileDto = {
        originalName: file.originalname,
        filename,
        filePath,
        checksum,
        mimeType: file.mimetype,
        fileSize: file.size,
        width,
        height,
        title: "",
        description: "",
        tags: [''],
        thumbnailFilename,
        thumbnailFilePath,
      };

      await this.createImageRecord(createFileDto);

      return {
        message: 'File uploaded successfully',
        filename,
        path: filePath,
        thumbnailPath: thumbnailFilePath,
      };
    } catch (error) {
      if (error instanceof ConflictException || error instanceof InternalServerErrorException) {
        throw error;
      }
      throw new InternalServerErrorException('Failed to process image upload');
    }
  }

  /**
   * Checks if file exists on filesystem
   */
  checkFileExists(filePath: string): boolean {
    return fs.existsSync(filePath);
  }

  /**
   * Gets file stream for serving
   */
  getFileStream(filePath: string): fs.ReadStream {
    return fs.createReadStream(filePath);
  }

  /**
   * Determines content type based on file extension
   */
  getContentType(filename: string): string {
    const ext = path.extname(filename).toLowerCase();
    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'image/jpeg';
      case '.png':
        return 'image/png';
      case '.gif':
        return 'image/gif';
      case '.webp':
        return 'image/webp';
      default:
        return 'application/octet-stream';
    }
  }

  /**
   * Cleanup - removes both original and thumbnail files
   */
  async deleteImageFiles(imageId: string, uploadPath: string): Promise<void> {
    try {
      const image = await this.prisma.image.findUnique({
        where: { id: imageId },
      });

      if (!image) return;

      // Delete original file
      const originalPath = path.join(uploadPath, image.filename);
      if (fs.existsSync(originalPath)) {
        await unlink(originalPath);
      }

      // Delete thumbnail file
      if (image.thubmnailFilename) {
        const thumbnailPath = path.join(uploadPath, 'thumbnails', image.thubmnailFilename);
        if (fs.existsSync(thumbnailPath)) {
          await unlink(thumbnailPath);
        }
      }

      // Delete database record
      await this.prisma.image.delete({
        where: { id: imageId },
      });
    } catch (error) {
      console.error('Failed to delete image files:', error);
      throw new InternalServerErrorException('Failed to delete image files');
    }
  }
}
