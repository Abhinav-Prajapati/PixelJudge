import { Injectable, ConflictException, InternalServerErrorException } from '@nestjs/common';
import { CreateFileDto } from './dto/create-file.dto';
import { UpdateFileDto } from './dto/update-file.dto';
import { PrismaService } from 'prisma/prisma.service';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';

const writeFile = promisify(fs.writeFile);
const unlink = promisify(fs.unlink);

@Injectable()
export class FilesService {
  constructor(private prisma: PrismaService) { }

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
          tags: createFileDto.tags || undefined, // Convert null to undefined for Prisma
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
      const filePath = path.join(uploadPath, filename);

      await writeFile(filePath, file.buffer);

      return {
        filename,
        filePath: `/library/${filename}`,
      };
    } catch (error) {
      throw new InternalServerErrorException('Failed to save file to filesystem');
    }
  }

  /**
   * Processes image upload with duplicate checking
   */
  async processImageUpload(
    file: Express.Multer.File,
    uploadPath: string
  ): Promise<{ message: string; filename: string; path: string; isExisting?: boolean }> {
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
          isExisting: true,
        };
      }

      // Save new file
      const { filename, filePath } = await this.saveFileToSystem(file, uploadPath);

      // Create image record
      const createFileDto: CreateFileDto = {
        originalName: file.originalname,
        filename,
        filePath,
        checksum,
        mimeType: file.mimetype,
        fileSize: file.size,
        width: null, // You might want to extract this using sharp or similar
        height: null,
        title: null,
        description: null,
        tags: null, // Changed to match DTO type
      };

      await this.createImageRecord(createFileDto);

      return {
        message: 'File uploaded successfully',
        filename,
        path: filePath,
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
}
