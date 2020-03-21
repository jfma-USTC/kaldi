// bin/align-equal.cc

// Copyright 2009-2013  Microsoft Corporation
//                      Johns Hopkins University (Author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
//#include "tree/context-dep.h"
#include "hmm/transition-model-2D.h"
//#include "fstext/fstext-lib.h"
//#include "decoder/training-graph-compiler.h"


/** @brief Write equally spaced alignments of utterances (to get training started).
*/
int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;
		//using fst::SymbolTable;
		//using fst::VectorFst;
		//using fst::StdArc;

		const char *usage = "Write equally spaced alignments of utterances "
			"(to get training started)\n"
			"Usage:  align-equal <model-in> <block-rspecifier> <transcriptions-rspecifier> "
			"<alignments-wspecifier>\n"
			"e.g.: \n"
			" align-equal 0.mdl ark:block 'ark:sym2int.pl -f 2- words.txt text|'"
			" ark,t:equal.ali\n";

		ParseOptions po(usage);
		//std::string disambig_rxfilename;
		//po.Register("read-disambig-syms", &disambig_rxfilename, "File containing list of disambiguation symbols in phone symbol table");
		po.Read(argc, argv);

		if (po.NumArgs() != 4) {
			po.PrintUsage();
			exit(1);
		}

		//std::string tree_in_filename = po.GetArg(1);
		//std::string lex_in_filename = po.GetArg(3);
		//std::string feature_rspecifier = po.GetArg(4);
		//std::string alignment_wspecifier = po.GetArg(6);

		std::string model_in_filename = po.GetArg(1); // .mdl��ҪΪ�˻�ȡstate-map�Լ���Ӧ��trans_state
		std::string block_rspecifier = po.GetArg(2); // ��ȡÿ��sample��block-map��Ϣ
		std::string transcript_rspecifier = po.GetArg(3); // ��ȡÿ��sample��phone����
		std::string alignment_wspecifier = po.GetArg(4); // д��ÿ��sample��Ӧ��alignment vector

		//ContextDependency ctx_dep;
		//ReadKaldiObject(tree_in_filename, &ctx_dep);

		TransitionModel_2D trans_model;
		{
			bool binary; // binary��ֵͨ��kiʵ������������ж���������Ϊ1���ı���ʽΪ0
			Input ki(model_in_filename, &binary); // Kaldi���ݿ�ͷ�Ƿ�"\0B"���ж��Ƕ����ƻ����ı���ʽ��ͬʱ׼����һ����ki
			trans_model.Read(ki.Stream(), binary);// ��.mdl�����ȡ����TransitionModel��Ķ���trans_model
		}
		const HmmTopology_2D topo = trans_model.GetTopo();

		/*
		 need VectorFst because we will change it by adding subseq symbol.
		VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_in_filename);

		TrainingGraphCompilerOptions gc_opts(1.0, true);  // true -> Dan style graph.

		std::vector<int32> disambig_syms;
		if (disambig_rxfilename != "")
		  if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
			KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
					  << disambig_rxfilename;


		TrainingGraphCompiler gc(trans_model,
								 ctx_dep,
								 lex_fst,
								 disambig_syms,
								 gc_opts);

		lex_fst = NULL;  // we gave ownership to gc.
		*/

		SequentialInt32VectorReader block_info_reader(block_rspecifier);
		RandomAccessInt32VectorReader transcript_reader(transcript_rspecifier);
		Int32VectorWriter alignment_writer(alignment_wspecifier);

		int32 done = 0, no_transcript = 0, other_error = 0;
		for (; !block_info_reader.Done(); block_info_reader.Next()) {
			std::string key = block_info_reader.Key();
			const std::vector<int32> &block_info = block_info_reader.Value();
			// ���ĳ��sample��block-information����������row_num col_num tol_num�����������+1���������sample
			if (block_info.size() != 3) {
				KALDI_WARN << "Block information size supposed to be 3, but get "
					<< block_info.size() << " from " << key << " instead.";
				other_error++;
				continue;
			}
			size_t block_row_num = static_cast<size_t>(block_info[0]),
				  block_col_num = static_cast<size_t>(block_info[1]),
				  block_tol_num = static_cast<size_t>(block_info[2]);
			// ���ĳ��sample��block-information����ȷ��row_num*col_num��=tol_num�����������+1���������sample
			if (block_tol_num != block_row_num * block_col_num) {
				KALDI_WARN << "Block rows number * cols number not equal to total number, sample name:" << key
					<< "rows:" << block_row_num << " cols:" << block_col_num << " total:" << block_tol_num;
				other_error++;
				continue;
			}
			// ������sample��text��Ӧ��phone��������д���
			if (transcript_reader.HasKey(key)) {
				const std::vector<int32> &transcript = transcript_reader.Value(key);
				//TODO:��ʱ�����ܳ���һ��phone��sample
				if (transcript.size() != 1) {
					KALDI_WARN << "Currently could only handle single phone for each sample, but get "
						<< transcript.size() << " phones for " << key;
					other_error++;
					continue;
				}
				int32 phone_id = transcript[0],//��������Ӧ��phone
					state_row_num = topo.TopologyShapeRowsForPhone(phone_id),//��phone��Ӧ��state-map������
					state_col_num = topo.TopologyShapeColsForPhone(phone_id),//��phone��Ӧ��state-map������
					hmm_state, //ĳһblock��Ӧ��state�ڸ�phone�е�index��0-based
					trans_state; //��state������state�е����
				//������sample��block-map����/������һ��С��state-map�����޷��ṩequal-ali-2D���������sample
				if (block_row_num < state_row_num || block_col_num < state_col_num) {
					KALDI_WARN << "Sample name: " << key << " corrspanding phone: " << phone_id
						<< " have state map for " << state_row_num << " rows " << state_col_num << " cols "
						<< "but only have " << block_row_num << " rows " << block_col_num << " cols frames, "
						<< "cannot create proper equal-alignment.";
						other_error++;
					continue;
				}
				size_t step_rows = static_cast<size_t>(floor(block_row_num / state_row_num)),//ÿstate-map�����ƶ�һ���Ӧ��block-map���ƶ��ĸ���
					step_cols = static_cast<size_t>(floor(block_col_num / state_col_num)),//ÿstate-map�����ƶ�һ���Ӧ��block-map���ƶ��ĸ���
					state_row, //state-map�е�row_index��0-based
					state_col; //state-map�е�col_index��0-based
				//������¼ÿ��block��Ӧ��trans_state��Ӧ��ϵ�������block�����������һ��vector
				std::vector<int32> blocks2states(block_tol_num);
				for (size_t row_i = 0; row_i < block_row_num; row_i++) {
					for (size_t col_j = 0; col_j < block_col_num; col_j++) {
						state_row = std::min(static_cast<size_t>(state_row_num), static_cast<size_t>(ceil((row_i + 1) / step_rows))) - 1;
						state_col = std::min(static_cast<size_t>(state_col_num), static_cast<size_t>(ceil((col_j + 1) / step_cols))) - 1;
						hmm_state = static_cast<int32>(state_row * state_col_num + state_col);
						trans_state = trans_model.PairToState(phone_id, hmm_state);
						blocks2states[row_i*block_col_num + col_j] = trans_state;
					}
				}
				alignment_writer.Write(key, blocks2states);
				done++;
			}
			else {
				KALDI_WARN << "AlignEqual2D: no blocks number info for utterance " << key;
				no_transcript++;
			}
		}
		if (done != 0 && no_transcript == 0 && other_error == 0) {
			KALDI_LOG << "Success: done " << done << " utterances.";
		}
		else {
			KALDI_WARN << "Computed " << done << " alignments; " << no_transcript
				<< " transcripts size not equal to 1, " << other_error
				<< " had other errors.";
		}
		if (done != 0) return 0;
		else return 1;
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}


