/*
  Warnings:

  - You are about to drop the `images` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "public"."images";

-- CreateTable
CREATE TABLE "images" (
    "id" TEXT NOT NULL,
    "originalName" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "checksum" TEXT NOT NULL,
    "mimeType" TEXT,
    "fileSize" INTEGER,
    "width" INTEGER,
    "height" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "title" TEXT,
    "description" TEXT,
    "tags" TEXT[],
    "thubmnailFilename" TEXT,
    "thumbnailFilePath" TEXT,
    "thubnailWidth" INTEGER,
    "thubnailHeight" INTEGER,

    CONSTRAINT "images_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "images_filename_key" ON "images"("filename");

-- CreateIndex
CREATE UNIQUE INDEX "images_checksum_key" ON "images"("checksum");

-- CreateIndex
CREATE UNIQUE INDEX "images_thubmnailFilename_key" ON "images"("thubmnailFilename");

-- CreateIndex
CREATE INDEX "images_checksum_idx" ON "images"("checksum");

-- CreateIndex
CREATE INDEX "images_createdAt_idx" ON "images"("createdAt");

-- CreateIndex
CREATE INDEX "images_tags_idx" ON "images"("tags");
