import {
  Controller,
  Get,
  Post,
  Delete,
  Param,
  UploadedFile,
  UseInterceptors,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import * as fs from 'fs';
import * as path from 'path';
import { extname } from 'path';

@Controller('files')
export class FilesController {
  private readonly uploadPath = path.join(__dirname, '../../images-library');

  // 📤 Upload a single file
  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: './images-library',
        filename: (req, file, cb) => {
          const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
          const ext = extname(file.originalname);
          cb(null, `${file.fieldname}-${uniqueSuffix}${ext}`);
        },
      }),
      fileFilter: (req, file, cb) => {
        if (!file.mimetype.match(/\/(jpg|jpeg|png|gif)$/)) {
          return cb(new Error('Only image files are allowed!'), false);
        }
        cb(null, true);
      },
    }),
  )
  uploadFile(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      throw new HttpException('No file uploaded', HttpStatus.BAD_REQUEST);
    }

    return {
      message: 'File uploaded successfully',
      filename: file.filename,
      path: `/images/${file.filename}`,
    };
  }

  // 📂 List all files
  @Get()
  findAll() {
    try {
      const files = fs.readdirSync(this.uploadPath);
      return files.map((filename) => ({
        filename,
        url: `/images/${filename}`,
      }));
    } catch (err) {
      throw new HttpException('Failed to read upload directory', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
}

