/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <luci/ImporterEx.h>
#include <luci/CircleOptimizer.h>
#include <luci/DynamicBatchToSingleBatch.h>
#include <luci/Service/ChangeOutputs.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "WeightDivider.h"
#include "WeightOnlyFormatFileCreator.h"

void add_switch(arser::Arser &arser, const char *opt, const char *desc)
{
  arser.add_argument(opt).nargs(0).default_value(false).help(desc);
}

bool store_into_file(char *buffer_data, const std::string &file_path, const size_t size)
{
  std::ofstream file(file_path, std::ios::binary);

  if (not file.is_open())
    return false;

  file.write(buffer_data, size);

  file.close();

  return true;
}

int entry(int argc, char **argv)
{
  // Simple argument parser (based on map)
  luci::CircleOptimizer optimizer;

  arser::Arser arser("circle-weight-divider divides weight from circle file to new file");

  arser::Helper::add_verbose(arser);

  //add_switch(arser, "--divide", "This will fold AddV2 operators with constant inputs");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output_circle").help("Output circle model");
  arser.add_argument("output_weight").help("Output weight model file");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  auto input_path = arser.get<std::string>("input");
  auto output_circle_path = arser.get<std::string>("output_circle");
  auto output_weight_path = arser.get<std::string>("output_weight");

  // Import from input Circle file
  luci::ImporterEx importerex;
  auto module_divider = importerex.importVerifyModule(input_path);
  if (module_divider == nullptr)
    return EXIT_FAILURE;

  auto module_creator = importerex.importVerifyModule(input_path);
  if (module_creator == nullptr)
    return EXIT_FAILURE;

  // divide weights from original circle model
  luci::WeightDivider divider(module_divider.get());
  if (not divider.divide())
  {
    std::cerr << "ERROR: Failed to divide consts from original circle: '" << input_path << "'" << std::endl;
    return 255;
  }

  // create weight only format file
  //luci::WeightOnlyFormatFileCreator creator(module_creator.get());
  //std::tuple<std::unique_ptr<char>, size_t> wof_data = creator.create();

  // Export to output Circle file
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(module_divider.get(), output_circle_path);
  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_circle_path << "'" << std::endl;
    return 255;
  }

//  // Export to output Weight Only Format file
//  if (not store_into_file(std::get<0>(wof_data).get(), output_weight_path, std::get<1>(wof_data)))
//  {
//    std::cerr << "ERROR: Failed to export '" << output_weight_path << "'" << std::endl;
//    return 255;
//  }

  return 0;
}
