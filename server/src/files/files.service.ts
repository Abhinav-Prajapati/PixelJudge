import { ConflictException, Injectable, InternalServerErrorException } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { PrismaService } from 'prisma/prisma.service';
import * as sharp from 'sharp';
import { promisify } from 'util';
import { CreateFileDto } from './dto/create-file.dto';

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
          thumbnailHeight: createFileDto.thumbnailHeight,
          thumbnailWidth: createFileDto.thumbnailWidth
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
        .resize({ height: 350 }) // Only set hight, auto width to maintain aspect ratio
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
   * Returns all images grouped by date with their original and thumbnail URLs
   */
  async getAllImagesWithUrls(baseUrl: string): Promise<{
    [date: string]: {
      id: string;
      originalName: string;
      originalUrl: string;
      thumbnailUrl: string | null;
      width: number | null;
      height: number | null;
      fileSize: number | null;
      mimeType: string | null;
      title: string | null;
      description: string | null;
      tags: string[];
      createdAt: Date;
      updatedAt: Date;
      thumbnailHeight: number | null;
      thumbnailWidth: number | null;
    }[];
  }> {
    try {
      const images = await this.prisma.image.findMany({
        orderBy: {
          createdAt: 'desc', // Most recent first
        },
      });

      // Group images by date (YYYY-MM-DD format)
      const groupedImages: { [date: string]: any[] } = {};

      images.forEach(image => {
        // Format date as YYYY-MM-DD
        const dateKey = image.createdAt.toISOString().split('T')[0];

        if (!groupedImages[dateKey]) {
          groupedImages[dateKey] = [];
        }

        groupedImages[dateKey].push({
          id: image.id,
          originalName: image.originalName,
          originalUrl: `${baseUrl}/files?path=${image.filePath}`,
          width: image.width,
          height: image.height,
          fileSize: image.fileSize,
          mimeType: image.mimeType,
          thumbnailUrl: image.thumbnailFilePath
            ? `${baseUrl}/files?path=${image.thumbnailFilePath}`
            : null,
          thumbnailHeight: image.thumbnailHeight,
          thumbnailWidth: image.thumbnailWidth,
          title: image.title,
          description: image.description,
          tags: image.tags,
          createdAt: image.createdAt,
          updatedAt: image.updatedAt,
        });
      });

      // Sort dates in descending order and create the final object
      const sortedDates = Object.keys(groupedImages).sort((a, b) => b.localeCompare(a));
      const result: { [date: string]: any[] } = {};

      sortedDates.forEach(date => {
        result[date] = groupedImages[date];
      });

      return result;
    } catch (error) {
      console.error('Failed to get images with URLs:', error);
      throw new InternalServerErrorException('Failed to retrieve images');
    }
  }

  /**
   * Returns paginated images with their original and thumbnail URLs
   */
  async getPaginatedImagesWithUrls(
    baseUrl: string,
    page: number = 1,
    limit: number = 20
  ): Promise<{
    images: {
      id: string;
      originalName: string;
      originalUrl: string;
      thumbnailUrl: string | null;
      width: number | null;
      height: number | null;
      fileSize: number | null;
      mimeType: string | null;
      title: string | null;
      description: string | null;
      tags: string[];
      createdAt: Date;
      updatedAt: Date;
    }[];
    pagination: {
      currentPage: number;
      totalPages: number;
      totalItems: number;
      hasNext: boolean;
      hasPrevious: boolean;
    };
  }> {
    try {
      const skip = (page - 1) * limit;

      // Get total count for pagination
      const totalItems = await this.prisma.image.count();
      const totalPages = Math.ceil(totalItems / limit);

      // Get paginated images
      const images = await this.prisma.image.findMany({
        skip,
        take: limit,
        orderBy: {
          createdAt: 'desc',
        },
      });

      const imagesWithUrls = images.map(image => ({
        id: image.id,
        originalName: image.originalName,
        originalUrl: `${baseUrl}/files?path=${image.filePath}`,
        thumbnailUrl: image.thumbnailFilePath
          ? `${baseUrl}/files?path=${image.thumbnailFilePath}`
          : null,
        width: image.width,
        height: image.height,
        fileSize: image.fileSize,
        mimeType: image.mimeType,
        title: image.title,
        description: image.description,
        tags: image.tags,
        createdAt: image.createdAt,
        updatedAt: image.updatedAt,
      }));

      return {
        images: imagesWithUrls,
        pagination: {
          currentPage: page,
          totalPages,
          totalItems,
          hasNext: page < totalPages,
          hasPrevious: page > 1,
        },
      };
    } catch (error) {
      console.error('Failed to get paginated images with URLs:', error);
      throw new InternalServerErrorException('Failed to retrieve paginated images');
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

      // Extract thumb metadata
      const thumbnailSize = await this.extractImageMetadata(thumbnailBuffer);

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
        thumbnailHeight: thumbnailSize.height,
        thumbnailWidth: thumbnailSize.width,
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
