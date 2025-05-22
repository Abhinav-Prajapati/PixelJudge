import {
  Controller,
  Get,
  Res,
  Post,
  Delete,
  Param,
  UploadedFile,
  NotFoundException,
  UseInterceptors,
  HttpException,
  HttpStatus,
  BadRequestException,
  Query,
  Req,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Response } from 'express';
import { FilesService } from './files.service';
import * as path from 'path';

// TODO: Move to environment variables
const uploadPath = path.join(process.cwd(), 'library/');

@Controller('files')
export class FilesController {
  constructor(private filesService: FilesService) { }

  /**
   * Uploads a single image file with duplicate checking
   */
  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      fileFilter: (req, file, cb) => {
        if (!file.mimetype.match(/\/(jpg|jpeg|png|gif|webp)$/)) {
          return cb(new Error('Only image files are allowed!'), false);
        }
        cb(null, true);
      },
      limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
      },
    }),
  )
  async uploadFile(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      throw new BadRequestException('No file uploaded');
    }

    try {
      const result = await this.filesService.processImageUpload(file, uploadPath);
      return result;
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        'Failed to upload file',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
   * Serves image files by filename
   */
  @Get()
  async serveImage(@Query('path') filename: string, @Res() res: Response) {
    try {
      const filePath = path.join(uploadPath, filename);
      console.log(filePath)

      if (!this.filesService.checkFileExists(filePath)) {
        throw new NotFoundException('Image not found');
      }

      // Set appropriate content-type header
      const contentType = this.filesService.getContentType(filename);
      res.setHeader('Content-Type', contentType);

      // Set cache headers for better performance
      res.setHeader('Cache-Control', 'public, max-age=31536000'); // 1 year
      res.setHeader('ETag', filename); // Simple ETag based on filename

      // Stream the file to the response
      const fileStream = this.filesService.getFileStream(filePath);
      fileStream.pipe(res);

      fileStream.on('error', () => {
        if (!res.headersSent) {
          throw new HttpException(
            'Error serving file',
            HttpStatus.INTERNAL_SERVER_ERROR,
          );
        }
      });
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        'Failed to serve image',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
 * Returns all images with their original and thumbnail URLs
 */
  @Get('images')
  async getAllImages(@Req() request: Request) {
    try {
      const baseUrl = `http://localhost:3000`
      const images = await this.filesService.getAllImagesWithUrls(baseUrl);
      return {
        success: true,
        data: images,
        total: images.length,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        'Failed to retrieve images',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
   * Returns paginated images with their original and thumbnail URLs
   */
  @Get('images/paginated')
  async getPaginatedImages(
    @Query('page') page: string = '1',
    @Query('limit') limit: string = '20',
    @Query('tags') tags: string = '',
    @Req() request: Request
  ) {
    try {
      const baseUrl = `http://localhost:3000`
      const pageNumber = parseInt(page, 10) || 1;
      const limitNumber = parseInt(limit, 10) || 20;

      // Validate pagination parameters
      if (pageNumber < 1) {
        throw new BadRequestException('Page number must be greater than 0');
      }
      if (limitNumber < 1 || limitNumber > 100) {
        throw new BadRequestException('Limit must be between 1 and 100');
      }

      let result = await this.filesService.getPaginatedImagesWithUrls(
        baseUrl,
        pageNumber,
        limitNumber
      );

      return {
        success: true,
        data: result.images,
        pagination: result.pagination,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        'Failed to retrieve paginated images',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  /**
   * Deletes an image file and its database record
   */
  @Delete(':filename')
  async deleteImage(@Param('filename') filename: string) {
    throw new HttpException('Not implemented yet', HttpStatus.NOT_IMPLEMENTED);
  }
}
