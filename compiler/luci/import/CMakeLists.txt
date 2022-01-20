file(GLOB SOURCES "src/*.cpp")
list(APPEND SOURCES
#        src/Nodes/CircleAbs.cpp
        src/Nodes/CircleAdd.cpp
#        src/Nodes/CircleAddN.cpp
#        src/Nodes/CircleArgMax.cpp
#        src/Nodes/CircleArgMin.cpp
        src/Nodes/CircleAveragePool2D.cpp
#        src/Nodes/CircleBatchMatMul.cpp
#        src/Nodes/CircleBatchToSpaceND.cpp
#        src/Nodes/CircleBCQFullyConnected.cpp
#        src/Nodes/CircleBCQGather.cpp
#        src/Nodes/CircleBidirectionalSequenceLSTM.cpp
#        src/Nodes/CircleCast.cpp
#        src/Nodes/CircleCeil.cpp
#        src/Nodes/CircleCustom.cpp
#        src/Nodes/CircleConcatenation.cpp
        src/Nodes/CircleConst.cpp
        src/Nodes/CircleConv2D.cpp
#        src/Nodes/CircleCos.cpp
#        src/Nodes/CircleDepthToSpace.cpp
        src/Nodes/CircleDepthwiseConv2D.cpp
        src/Nodes/CircleDequantize.cpp
#        src/Nodes/CircleDiv.cpp
#        src/Nodes/CircleElu.cpp
#        src/Nodes/CircleEqual.cpp
#        src/Nodes/CircleExp.cpp
#        src/Nodes/CircleExpandDims.cpp
#        src/Nodes/CircleFakeQuant.cpp
#        src/Nodes/CircleFill.cpp
#        src/Nodes/CircleFloor.cpp
#        src/Nodes/CircleFloorDiv.cpp
#        src/Nodes/CircleFloorMod.cpp
        src/Nodes/CircleFullyConnected.cpp
#        src/Nodes/CircleGather.cpp
#        src/Nodes/CircleGatherNd.cpp
#        src/Nodes/CircleGreater.cpp
#        src/Nodes/CircleGreaterEqual.cpp
#        src/Nodes/CircleIf.cpp
#        src/Nodes/CircleInstanceNorm.cpp
#        src/Nodes/CircleL2Normalize.cpp
#        src/Nodes/CircleL2Pool2D.cpp
#        src/Nodes/CircleLeakyRelu.cpp
#        src/Nodes/CircleLess.cpp
#        src/Nodes/CircleLessEqual.cpp
#        src/Nodes/CircleLocalResponseNormalization.cpp
#        src/Nodes/CircleLog.cpp
#        src/Nodes/CircleLogicalAnd.cpp
#        src/Nodes/CircleLogicalNot.cpp
#        src/Nodes/CircleLogicalOr.cpp
#        src/Nodes/CircleLogistic.cpp
#        src/Nodes/CircleLogSoftmax.cpp
#        src/Nodes/CircleMatrixDiag.cpp
#        src/Nodes/CircleMatrixSetDiag.cpp
#        src/Nodes/CircleMaximum.cpp
        src/Nodes/CircleMaxPool2D.cpp
#        src/Nodes/CircleMean.cpp
#        src/Nodes/CircleMinimum.cpp
#        src/Nodes/CircleMirrorPad.cpp
        src/Nodes/CircleMul.cpp
#        src/Nodes/CircleNeg.cpp
#        src/Nodes/CircleNonMaxSuppressionV4.cpp
#        src/Nodes/CircleNonMaxSuppressionV5.cpp
#        src/Nodes/CircleNotEqual.cpp
#        src/Nodes/CircleOneHot.cpp
#        src/Nodes/CirclePack.cpp
#        src/Nodes/CirclePad.cpp
#        src/Nodes/CirclePadV2.cpp
#        src/Nodes/CirclePow.cpp
#        src/Nodes/CirclePRelu.cpp
        src/Nodes/CircleQuantize.cpp
#        src/Nodes/CircleRange.cpp
#        src/Nodes/CircleRank.cpp
#        src/Nodes/CircleReduceAny.cpp
#        src/Nodes/CircleReduceMax.cpp
#        src/Nodes/CircleReduceMin.cpp
#        src/Nodes/CircleReduceProd.cpp
#        src/Nodes/CircleRelu.cpp
#        src/Nodes/CircleRelu6.cpp
#        src/Nodes/CircleReluN1To1.cpp
        src/Nodes/CircleReshape.cpp
#        src/Nodes/CircleResizeBilinear.cpp
#        src/Nodes/CircleResizeNearestNeighbor.cpp
#        src/Nodes/CircleReverseSequence.cpp
#        src/Nodes/CircleReverseV2.cpp
#        src/Nodes/CircleRound.cpp
#        src/Nodes/CircleRsqrt.cpp
#        src/Nodes/CircleScatterNd.cpp
#        src/Nodes/CircleSegmentSum.cpp
#        src/Nodes/CircleSelect.cpp
#        src/Nodes/CircleSelectV2.cpp
#        src/Nodes/CircleShape.cpp
#        src/Nodes/CircleSin.cpp
#        src/Nodes/CircleSlice.cpp
        src/Nodes/CircleSoftmax.cpp
#        src/Nodes/CircleSpaceToBatchND.cpp
#        src/Nodes/CircleSpaceToDepth.cpp
#        src/Nodes/CircleSparseToDense.cpp
#        src/Nodes/CircleSplit.cpp
#        src/Nodes/CircleSplitV.cpp
#        src/Nodes/CircleSqrt.cpp
#        src/Nodes/CircleSquare.cpp
#        src/Nodes/CircleSquaredDifference.cpp
#        src/Nodes/CircleSqueeze.cpp
#        src/Nodes/CircleStridedSlice.cpp
#        src/Nodes/CircleSub.cpp
#        src/Nodes/CircleSum.cpp
#        src/Nodes/CircleTanh.cpp
#        src/Nodes/CircleTile.cpp
#        src/Nodes/CircleTopKV2.cpp
#        src/Nodes/CircleTranspose.cpp
#        src/Nodes/CircleTransposeConv.cpp
#        src/Nodes/CircleUnidirectionalSequenceLSTM.cpp
#        src/Nodes/CircleUnique.cpp
#        src/Nodes/CircleUnpack.cpp
#        src/Nodes/CircleWhere.cpp
#        src/Nodes/CircleWhile.cpp
#        src/Nodes/CircleZerosLike.cpp
        )
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

if (NOT LUCI_LIBRARY_TYPE)
  set(LUCI_LIBRARY_TYPE "SHARED")
endif(NOT LUCI_LIBRARY_TYPE)

add_library(luci_import ${LUCI_LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_import PRIVATE src)
target_include_directories(luci_import PUBLIC include)
target_link_libraries(luci_import PUBLIC luci_lang)
target_link_libraries(luci_import PUBLIC luci_profile)
target_link_libraries(luci_import PUBLIC luci_plan)
target_link_libraries(luci_import PUBLIC mio_circle)
target_link_libraries(luci_import PRIVATE luci_env)
target_link_libraries(luci_import PRIVATE luci_log)
target_link_libraries(luci_import PRIVATE luci_logex)
target_link_libraries(luci_import PRIVATE nncc_common)
target_link_libraries(luci_import PRIVATE locop)
target_link_libraries(luci_import PRIVATE oops)
install(TARGETS luci_import DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_import_test ${TESTS})
target_include_directories(luci_import_test PRIVATE src)
target_link_libraries(luci_import_test luci_import)
target_link_libraries(luci_import_test oops)
target_link_libraries(luci_import_test luci_plan)
target_link_libraries(luci_import_test luci_lang)
target_link_libraries(luci_import_test mio_circle)
