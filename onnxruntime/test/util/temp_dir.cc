// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/temp_dir.h"

#include "gtest/gtest.h"

#include "core/platform/env.h"

namespace onnxruntime {
namespace test {

TemporaryDirectory::TemporaryDirectory(const PathString& path)
    : path_{path} {
  // EXPECT and throw to fail even if anyone is catching exceptions
  const bool exists = PlatformApi::FolderExists(path_);
  EXPECT_TRUE(!exists) << "Temporary directory " << path_ << " already exists.";
  ORT_ENFORCE(!exists);
  const auto status = PlatformApi::CreateFolder(path_);
  EXPECT_TRUE(status.IsOK()) << "Failed to create temporary directory " << path_ << ": " << status.ErrorMessage();
  ORT_ENFORCE(status.IsOK());
}

TemporaryDirectory::~TemporaryDirectory() {
  const auto status = PlatformApi::DeleteFolder(path_);
  EXPECT_TRUE(status.IsOK()) << "Failed to delete temporary directory " << path_ << ": " << status.ErrorMessage();
}

}  // namespace test
}  // namespace onnxruntime
