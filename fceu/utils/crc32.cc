/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *  Copyright (C) 2002 Xodnizel
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include "fceu/types.h"
#include "fceu/utils/crc32.h"

#include <zlib.h>
uint32 CalcCRC32(uint32 crc, uint8 *buf, uint32 len)
{
 return(crc32(crc,buf,len));
}

uint32 FCEUI_CRC32(uint32 crc, uint8 *buf, uint32 len)
{
 return(CalcCRC32(crc,buf,len));
}
