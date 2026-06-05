#pragma once

/**
 * storage.h — SD card batched writer + EEPROM session counter.
 *
 * Packets are queued in a RAM batch of SD_BATCH_SIZE entries and written to
 * the card in a single block operation.  A periodic flush (SD_FLUSH_INTERVAL)
 * calls File::flush() to commit the FAT metadata.
 *
 * Session IDs survive power-loss via EEPROM.  A magic word detects first-boot
 * so the counter initialises cleanly without garbage.
 */

#include <Arduino.h>
#include <SD.h>
#include <EEPROM.h>

#include "config.h"
#include "packet.h"

namespace storage_detail {
    File     _file;
    LogPacket _batch[SD_BATCH_SIZE];
    size_t    _batchCount = 0;
} // namespace storage_detail

// ── EEPROM session counter ────────────────────────────────────

/**
 * Read the last session ID from EEPROM, increment it, persist, and return
 * the new value.  On first boot (no magic word) returns 1.
 */
uint32_t storageLoadSession() {
    uint32_t magic, id;
    EEPROM.get(EEPROM_MAGIC_ADDR, magic);

    if (magic != SESSION_MAGIC) {
        id = 1;
        EEPROM.put(EEPROM_MAGIC_ADDR, SESSION_MAGIC);
        Serial.println("[EEPROM] First boot → session 1.");
    } else {
        EEPROM.get(EEPROM_ADDR, id);
        id++;
        Serial.printf("[EEPROM] Session %lu → %lu\n", id - 1, id);
    }

    EEPROM.put(EEPROM_ADDR, id);
    return id;
}

// ── SD init ───────────────────────────────────────────────────

/**
 * Mount the SD card and open (or create) the binary log file for session `id`.
 * Returns true on success.  Retries up to 5 times with 100 ms delay.
 */
bool storageInit(uint32_t sessionID) {
    bool sdOk = false;
    for (int i = 0; i < 5 && !sdOk; i++) {
        delay(100);
        sdOk = SD.begin(BUILTIN_SDCARD);
    }

    if (!sdOk) {
        Serial.println("[SD] ERROR: card not found.");
        return false;
    }
    Serial.println("[SD] Card OK.");

    char path[32];
    snprintf(path, sizeof(path), "/session_%05lu.bin", sessionID);
    storage_detail::_file = SD.open(path, FILE_WRITE);

    if (!storage_detail::_file) {
        Serial.printf("[SD] ERROR: cannot open %s\n", path);
        return false;
    }

    Serial.printf("[SD] %s  (%u bytes/packet, batch=%u)\n",
                  path, sizeof(LogPacket), SD_BATCH_SIZE);
    return true;
}

// ── Write helpers ─────────────────────────────────────────────

/**
 * Flush the RAM batch to the SD card.
 * If `fsync` is true, also call File::flush() to commit FAT metadata.
 */
void storageFlush(bool fsync = false) {
    if (storage_detail::_batchCount == 0) return;

    storage_detail::_file.write(
        reinterpret_cast<const uint8_t*>(storage_detail::_batch),
        storage_detail::_batchCount * sizeof(LogPacket)
    );
    storage_detail::_batchCount = 0;

    if (fsync) storage_detail::_file.flush();
}

/**
 * Enqueue one packet in the RAM batch.
 * Automatically calls storageFlush() (without fsync) when the batch is full.
 */
void storageWrite(const LogPacket& pkt) {
    storage_detail::_batch[storage_detail::_batchCount++] = pkt;
    if (storage_detail::_batchCount >= SD_BATCH_SIZE)
        storageFlush(false);
}