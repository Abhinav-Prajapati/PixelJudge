/*
 * TODO: make sure not to save duplicate image []
 * */

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
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { createReadStream, existsSync } from 'fs';
import { extname, join } from 'path';
import * as path from 'path';
import {Response} from 'express';

const uploadPath = path.join(process.cwd(), 'library');

@Controller('files')
export class FilesController {

  // 📤 Upload a single file
  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: uploadPath,
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
      path: `/library/${file.filename}`,
    };
  }

  @Get(':filename')
  serveImage(@Param('filename') filename: string, @Res() res: Response){
    const filePath = join(uploadPath, filename)
    if (!existsSync(filePath)) {
      throw new NotFoundException('Image not found');
    }

    // Set appropriate content-type header based on file extension (basic example)
    if (filename.endsWith('.jpg') || filename.endsWith('.jpeg')) {
      res.setHeader('Content-Type', 'image/jpeg');
    } else if (filename.endsWith('.png')) {
      res.setHeader('Content-Type', 'image/png');
    } else if (filename.endsWith('.gif')) {
      res.setHeader('Content-Type', 'image/gif');
    } else {
      res.setHeader('Content-Type', 'application/octet-stream');
    }
    // Stream the file to the response
    const fileStream = createReadStream(filePath);
    fileStream.pipe(res);
  }
}

